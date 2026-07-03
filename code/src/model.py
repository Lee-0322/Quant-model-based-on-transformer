from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SequenceRegressionDataset(Dataset):
    """单股票序列回归数据集。"""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray, indices: Optional[np.ndarray] = None) -> None:
        self.sequences = sequences
        self.labels = labels
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        if self.indices is None:
            return int(self.sequences.shape[0])
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        real_index = int(index if self.indices is None else self.indices[index])
        sequence = self.sequences[real_index]
        label = self.labels[real_index]

        if sequence.dtype != np.float32:
            sequence = sequence.astype(np.float32, copy=False)
        if not sequence.flags['C_CONTIGUOUS']:
            sequence = np.ascontiguousarray(sequence)
        sequence_tensor = torch.from_numpy(sequence)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        return sequence_tensor, label_tensor, real_index


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        real_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=real_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(inputs)
        last_hidden = encoded[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


@dataclass
class TrainingResult:
    best_state_dict: dict[str, torch.Tensor]
    best_score: float
    history: list[dict[str, float]]


def mean_daily_topk_return(
    predictions: np.ndarray,
    labels: np.ndarray,
    meta_df: pd.DataFrame,
    top_k: int = 5,
) -> float:
    """按日期取预测分数最高的前 k 只股票，计算平均真实收益。"""
    eval_df = meta_df.copy()
    eval_df['prediction'] = predictions.astype(float)
    eval_df['label'] = labels.astype(float)

    daily_returns: list[float] = []
    for _, group in eval_df.groupby('日期'):
        selected = group.nlargest(min(top_k, len(group)), 'prediction')
        if selected.empty:
            continue
        daily_returns.append(float(selected['label'].mean()))

    return float(np.mean(daily_returns)) if daily_returns else 0.0


def evaluate_torch_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    meta_df: pd.DataFrame,
    top_k: int,
) -> tuple[float, float]:
    """返回验证集 MSE 和按天 top-k 平均收益。"""
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    mse_terms: list[float] = []

    with torch.no_grad():
        for sequences, target, _ in tqdm(dataloader, desc='验证进度', leave=False):
            sequences = sequences.to(device)
            target = target.to(device)
            output = model(sequences)
            mse = torch.mean((output - target) ** 2).item()
            mse_terms.append(mse)
            preds.append(output.detach().cpu().numpy())
            labels.append(target.detach().cpu().numpy())

    pred_array = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    label_array = np.concatenate(labels) if labels else np.array([], dtype=np.float32)
    score = mean_daily_topk_return(pred_array, label_array, meta_df, top_k=top_k)
    return float(np.mean(mse_terms) if mse_terms else 0.0), score


def train_torch_model(
    model: nn.Module,
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    val_sequences: np.ndarray,
    val_labels: np.ndarray,
    val_meta: pd.DataFrame,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    top_k: int,
    device: torch.device,
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None,
) -> TrainingResult:
    """统一的 PyTorch 训练入口，按验证集收益保存最优模型。"""
    train_loader = DataLoader(
        SequenceRegressionDataset(train_sequences, train_labels, indices=train_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceRegressionDataset(val_sequences, val_labels, indices=val_indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state_dict = copy.deepcopy(model.state_dict())
    best_score = float('-inf')
    no_improve_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        total_batches = len(train_loader)
        tqdm.write(f'开始第 {epoch + 1}/{epochs} 个 epoch，本轮共 {total_batches} 个 batch。')
        train_progress = tqdm(
            train_loader,
            total=total_batches,
            desc=f'Epoch {epoch + 1}/{epochs}',
            unit='batch',
            leave=True,
        )
        for batch_index, (sequences, target, _) in enumerate(train_progress, start=1):
            sequences = sequences.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(sequences)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_progress.set_postfix({
                'batch': f'{batch_index}/{total_batches}',
                'loss': f'{loss.item():.6f}',
            })

        val_mse, val_score = evaluate_torch_model(model, val_loader, device, val_meta, top_k=top_k)
        epoch_info = {
            'epoch': float(epoch + 1),
            'train_mse': float(np.mean(train_losses) if train_losses else 0.0),
            'val_mse': float(val_mse),
            'val_topk_return': float(val_score),
        }
        history.append(epoch_info)
        tqdm.write(
            f"第 {epoch + 1}/{epochs} 轮完成 | "
            f"train_mse={epoch_info['train_mse']:.6f} | "
            f"val_mse={epoch_info['val_mse']:.6f} | "
            f"val_topk_return={epoch_info['val_topk_return']:.6f}"
        )

        if val_score > best_score:
            best_score = float(val_score)
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            tqdm.write(f'验证收益未提升，已连续 {no_improve_epochs} 轮。')

        if no_improve_epochs >= patience:
            tqdm.write(f'触发早停，连续 {patience} 轮没有提升。')
            break

    return TrainingResult(best_state_dict=best_state_dict, best_score=best_score, history=history)
