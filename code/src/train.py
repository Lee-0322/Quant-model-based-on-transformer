from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main() -> None:
    from config import SETTINGS
    from dataset import prepare_dataset_bundle, save_bundle_metadata, save_scaler
    from model import LSTMRegressor, train_torch_model

    set_seed(SETTINGS['seed'])
    print('开始准备 LSTM 数据集...')

    cache_dir = SETTINGS['paths'].get('cache_dir')
    bundle = prepare_dataset_bundle(
        data_file=SETTINGS['paths']['train_data'],
        sequence_length=SETTINGS['sequence_length'],
        feature_set=SETTINGS['feature_set'],
        horizon=SETTINGS['score_horizon'],
        num_processes=SETTINGS['num_processes'],
        val_ratio=SETTINGS['val_ratio'],
        min_val_days=SETTINGS['min_val_days'],
        cache_dir=cache_dir,
    )

    device = resolve_device()
    print(f'数据准备完成，当前设备: {device}')
    print(
        f"训练样本数: {len(bundle['train_sequences'])} | "
        f"验证样本数: {len(bundle['val_sequences'])} | "
        f"特征维度: {bundle['train_sequences'].shape[-1]}"
    )
    model = LSTMRegressor(
        input_dim=bundle['train_sequences'].shape[-1],
        hidden_dim=SETTINGS['model']['hidden_dim'],
        num_layers=SETTINGS['model']['num_layers'],
        dropout=SETTINGS['model']['dropout'],
    ).to(device)

    print('开始进入 LSTM 训练阶段...')
    result = train_torch_model(
        model=model,
        train_sequences=bundle['train_sequences'],
        train_labels=bundle['train_labels'],
        val_sequences=bundle['val_sequences'],
        val_labels=bundle['val_labels'],
        val_meta=bundle['val_meta'],
        batch_size=SETTINGS['train']['batch_size'],
        epochs=SETTINGS['train']['epochs'],
        learning_rate=SETTINGS['train']['learning_rate'],
        weight_decay=SETTINGS['train']['weight_decay'],
        patience=SETTINGS['train']['patience'],
        top_k=SETTINGS['top_k'],
        device=device,
    )

    SETTINGS['paths']['model_file'].parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'state_dict': result.best_state_dict,
            'input_dim': bundle['train_sequences'].shape[-1],
            'model_config': SETTINGS['model'],
        },
        SETTINGS['paths']['model_file'],
    )
    save_scaler(SETTINGS['paths']['scaler_file'], bundle['scaler'])
    save_bundle_metadata(
        SETTINGS['paths']['metadata_file'],
        {
            'model_name': 'lstm',
            'validation_topk_return': result.best_score,
            'split_date': bundle['split_date'],
            'feature_set': SETTINGS['feature_set'],
            'sequence_length': SETTINGS['sequence_length'],
            'score_horizon': SETTINGS['score_horizon'],
            'train_config': SETTINGS['train'],
            'model_config': SETTINGS['model'],
            'history': result.history,
        },
    )
    print(json.dumps({'validation_topk_return': result.best_score, 'epochs_ran': len(result.history)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
