from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from .utils import engineer_features_39, engineer_features_158plus39
except ImportError:
    from utils import engineer_features_39, engineer_features_158plus39


FEATURE_COLUMNS_MAP = {
    '39': [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
    ],
    '158+39': [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
        'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
        'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
        'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
        'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
        'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
        'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
        'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
        'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
        'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
        'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
        'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
    ],
}

FEATURE_ENGINEER_MAP = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
}


def load_market_data(csv_path: str | Path) -> pd.DataFrame:
    """读取原始行情数据，并统一股票代码与日期字段。"""
    data = pd.read_csv(csv_path, dtype={'股票代码': str})
    data['股票代码'] = data['股票代码'].astype(str).str.zfill(6)
    data['日期'] = pd.to_datetime(data['日期'])
    return data.sort_values(['股票代码', '日期']).reset_index(drop=True)


def _run_feature_engineering(group: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return FEATURE_ENGINEER_MAP[feature_set](group)


def build_feature_frame(
    raw_df: pd.DataFrame,
    feature_set: str = '158+39',
    num_processes: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """对全部股票做特征工程，并返回可直接建模的特征表。"""
    if feature_set not in FEATURE_ENGINEER_MAP:
        raise ValueError(f'不支持的特征集合: {feature_set}')

    groups = [group.copy() for _, group in raw_df.groupby('股票代码', sort=False)]
    if not groups:
        raise ValueError('输入数据为空，无法做特征工程')

    if num_processes and num_processes > 1:
        workers = min(num_processes, mp.cpu_count())
        worker = partial(_run_feature_engineering, feature_set=feature_set)
        with mp.Pool(processes=workers) as pool:
            feature_frames = list(
                tqdm(
                    pool.imap(worker, groups),
                    total=len(groups),
                    desc='特征工程',
                )
            )
    else:
        feature_frames = [
            _run_feature_engineering(group, feature_set)
            for group in tqdm(groups, total=len(groups), desc='特征工程')
        ]

    processed = pd.concat(feature_frames, ignore_index=True)
    processed['日期'] = pd.to_datetime(processed['日期'])
    processed = processed.sort_values(['股票代码', '日期']).reset_index(drop=True)

    feature_columns = FEATURE_COLUMNS_MAP[feature_set]
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return processed, feature_columns


def append_target_columns(feature_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """按照比赛口径构造 T+1 买入、T+5 卖出的收益率标签。"""
    output = feature_df.copy()
    output['buy_open'] = output.groupby('股票代码')['开盘'].shift(-1)
    output['sell_open'] = output.groupby('股票代码')['开盘'].shift(-horizon)
    output['label'] = (output['sell_open'] - output['buy_open']) / (output['buy_open'] + 1e-12)
    output['buy_date'] = output.groupby('股票代码')['日期'].shift(-1)
    output['sell_date'] = output.groupby('股票代码')['日期'].shift(-horizon)
    return output


def build_supervised_sequences(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = 60,
    horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """将逐日特征表转换成单股票监督学习样本。"""
    labeled = append_target_columns(feature_df, horizon=horizon)
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    meta_rows: list[dict[str, Any]] = []

    for stock_id, group in tqdm(labeled.groupby('股票代码', sort=False), desc='构建序列样本'):
        group = group.sort_values('日期').reset_index(drop=True)
        feature_values = group[feature_columns].to_numpy(dtype=np.float32)
        for idx in range(sequence_length - 1, len(group)):
            row = group.iloc[idx]
            if pd.isna(row['label']) or pd.isna(row['buy_date']) or pd.isna(row['sell_date']):
                continue
            sequence = feature_values[idx - sequence_length + 1: idx + 1]
            if sequence.shape[0] != sequence_length:
                continue
            sequences.append(sequence)
            labels.append(float(row['label']))
            meta_rows.append(
                {
                    'stock_id': stock_id,
                    '日期': row['日期'],
                    '买入日': row['buy_date'],
                    '卖出日': row['sell_date'],
                }
            )

    if not sequences:
        raise ValueError('没有构造出任何监督学习样本，请检查数据量与 sequence_length')

    meta_df = pd.DataFrame(meta_rows)
    meta_df['日期'] = pd.to_datetime(meta_df['日期'])
    meta_df['买入日'] = pd.to_datetime(meta_df['买入日'])
    meta_df['卖出日'] = pd.to_datetime(meta_df['卖出日'])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.float32), meta_df


def split_train_val_by_date(
    meta_df: pd.DataFrame,
    val_ratio: float = 0.15,
    min_val_days: int = 20,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """按日期做时间切分，避免未来信息泄露。"""
    unique_dates = np.array(sorted(meta_df['日期'].unique()))
    if unique_dates.size < 2:
        raise ValueError('可用日期太少，无法切分训练集和验证集')

    val_days = max(min_val_days, math.ceil(unique_dates.size * val_ratio))
    val_days = min(val_days, unique_dates.size - 1)
    split_date = pd.Timestamp(unique_dates[-val_days])
    train_mask = meta_df['日期'] < split_date
    val_mask = ~train_mask

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError('训练/验证切分失败，请调整 val_ratio 或最小验证天数')

    return train_mask.to_numpy(), val_mask.to_numpy(), split_date


def fit_sequence_scaler(train_sequences: np.ndarray) -> StandardScaler:
    """将序列样本按时间步展平后拟合标准化器。"""
    scaler = StandardScaler()
    flat = train_sequences.reshape(-1, train_sequences.shape[-1])
    scaler.fit(flat)
    return scaler


def transform_sequences(sequences: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """对序列样本应用与训练集一致的特征标准化。"""
    flat = sequences.reshape(-1, sequences.shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(sequences.shape).astype(np.float32)


def flatten_sequences(sequences: np.ndarray) -> np.ndarray:
    return sequences.reshape(sequences.shape[0], -1)


def _cache_dir_for_dataset(
    data_file: str | Path,
    feature_set: str,
    sequence_length: int,
    horizon: int,
    val_ratio: float,
    min_val_days: int,
    cache_base: Path,
) -> Path:
    """生成样本缓存目录，避免重复构建超大序列。"""
    data_path = Path(data_file).resolve()
    stat = data_path.stat()
    key_source = '|'.join(
        [
            str(data_path),
            str(int(stat.st_mtime)),
            feature_set,
            str(sequence_length),
            str(horizon),
            str(val_ratio),
            str(min_val_days),
        ]
    )
    key = hashlib.md5(key_source.encode('utf-8')).hexdigest()[:16]
    return cache_base / f'{data_path.stem}_{feature_set}_{key}'


def _count_supervised_samples(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    horizon: int,
) -> int:
    """第一遍扫描只统计样本数，便于按准确形状创建 memmap 文件。"""
    labeled = append_target_columns(feature_df, horizon=horizon)
    sample_count = 0
    for _, group in tqdm(labeled.groupby('股票代码', sort=False), desc='统计样本数'):
        group = group.sort_values('日期').reset_index(drop=True)
        for idx in range(sequence_length - 1, len(group)):
            row = group.iloc[idx]
            if pd.isna(row['label']) or pd.isna(row['buy_date']) or pd.isna(row['sell_date']):
                continue
            sample_count += 1
    return sample_count


def _build_supervised_memmap(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    horizon: int,
    output_dir: Path,
    storage_dtype: np.dtype,
) -> tuple[Path, Path, pd.DataFrame]:
    """两遍扫描构建监督样本，并将序列/标签写入 memmap 以降低内存峰值。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_samples = _count_supervised_samples(
        feature_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        horizon=horizon,
    )
    if total_samples <= 0:
        raise ValueError('没有构造出任何监督学习样本，请检查数据量与 sequence_length')

    sequence_path = output_dir / 'sequences.npy'
    labels_path = output_dir / 'labels.npy'
    stock_path = output_dir / 'meta_stock.npy'
    date_path = output_dir / 'meta_date_ns.npy'
    buy_path = output_dir / 'meta_buy_ns.npy'
    sell_path = output_dir / 'meta_sell_ns.npy'

    sequences_mm = np.lib.format.open_memmap(
        sequence_path,
        mode='w+',
        dtype=storage_dtype,
        shape=(total_samples, sequence_length, len(feature_columns)),
    )
    labels_mm = np.lib.format.open_memmap(labels_path, mode='w+', dtype=np.float32, shape=(total_samples,))
    stock_mm = np.lib.format.open_memmap(stock_path, mode='w+', dtype='U6', shape=(total_samples,))
    date_mm = np.lib.format.open_memmap(date_path, mode='w+', dtype=np.int64, shape=(total_samples,))
    buy_mm = np.lib.format.open_memmap(buy_path, mode='w+', dtype=np.int64, shape=(total_samples,))
    sell_mm = np.lib.format.open_memmap(sell_path, mode='w+', dtype=np.int64, shape=(total_samples,))

    labeled = append_target_columns(feature_df, horizon=horizon)
    cursor = 0
    for stock_id, group in tqdm(labeled.groupby('股票代码', sort=False), desc='写入监督样本'):
        group = group.sort_values('日期').reset_index(drop=True)
        feature_values = group[feature_columns].to_numpy(dtype=np.float32)
        date_values = group['日期'].to_numpy(dtype='datetime64[ns]')
        buy_values = group['buy_date'].to_numpy(dtype='datetime64[ns]')
        sell_values = group['sell_date'].to_numpy(dtype='datetime64[ns]')
        label_values = group['label'].to_numpy(dtype=np.float32)

        for idx in range(sequence_length - 1, len(group)):
            if np.isnan(label_values[idx]) or np.isnat(buy_values[idx]) or np.isnat(sell_values[idx]):
                continue
            seq = feature_values[idx - sequence_length + 1: idx + 1]
            if seq.shape[0] != sequence_length:
                continue

            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            sequences_mm[cursor] = seq.astype(storage_dtype, copy=False)
            labels_mm[cursor] = float(label_values[idx])
            stock_mm[cursor] = str(stock_id)
            date_mm[cursor] = date_values[idx].astype('int64')
            buy_mm[cursor] = buy_values[idx].astype('int64')
            sell_mm[cursor] = sell_values[idx].astype('int64')
            cursor += 1

    if cursor != total_samples:
        raise RuntimeError(f'样本写入计数不一致: expected={total_samples}, got={cursor}')

    sequences_mm.flush()
    labels_mm.flush()
    stock_mm.flush()
    date_mm.flush()
    buy_mm.flush()
    sell_mm.flush()

    meta_df = pd.DataFrame(
        {
            'stock_id': np.asarray(stock_mm),
            '日期': pd.to_datetime(np.asarray(date_mm), unit='ns'),
            '买入日': pd.to_datetime(np.asarray(buy_mm), unit='ns'),
            '卖出日': pd.to_datetime(np.asarray(sell_mm), unit='ns'),
        }
    )

    del sequences_mm, labels_mm, stock_mm, date_mm, buy_mm, sell_mm
    gc.collect()
    return sequence_path, labels_path, meta_df


def _fit_scaler_from_memmap(
    sequence_path: Path,
    train_indices: np.ndarray,
    feature_dim: int,
    chunk_samples: int = 2048,
) -> StandardScaler:
    """对训练子集分块拟合标准化器，避免一次性拉起全部样本。"""
    scaler = StandardScaler()
    seq_mm = np.load(sequence_path, mmap_mode='r')
    for start in tqdm(range(0, len(train_indices), chunk_samples), desc='拟合标准化器'):
        batch_idx = train_indices[start: start + chunk_samples]
        batch = np.asarray(seq_mm[batch_idx], dtype=np.float32)
        batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        scaler.partial_fit(batch.reshape(-1, feature_dim))
    del seq_mm
    gc.collect()
    return scaler


def _transform_subset_to_memmap(
    sequence_path: Path,
    scaler: StandardScaler,
    subset_indices: np.ndarray,
    output_path: Path,
    sequence_length: int,
    feature_dim: int,
    output_dtype: np.dtype,
    chunk_samples: int = 1024,
) -> Path:
    """将指定子集按分块标准化并写入新 memmap 文件。"""
    seq_mm = np.load(sequence_path, mmap_mode='r')
    out_mm = np.lib.format.open_memmap(
        output_path,
        mode='w+',
        dtype=output_dtype,
        shape=(len(subset_indices), sequence_length, feature_dim),
    )

    for start in tqdm(range(0, len(subset_indices), chunk_samples), desc=f'写入 {output_path.stem}'):
        batch_idx = subset_indices[start: start + chunk_samples]
        raw_batch = np.asarray(seq_mm[batch_idx], dtype=np.float32)
        raw_batch = np.nan_to_num(raw_batch, nan=0.0, posinf=0.0, neginf=0.0)
        flat = raw_batch.reshape(-1, feature_dim)
        scaled = scaler.transform(flat).reshape(raw_batch.shape).astype(output_dtype, copy=False)
        out_mm[start: start + len(batch_idx)] = scaled

    out_mm.flush()
    del out_mm, seq_mm
    gc.collect()
    return output_path


def prepare_dataset_bundle(
    data_file: str | Path,
    sequence_length: int,
    feature_set: str,
    horizon: int,
    num_processes: int,
    val_ratio: float,
    min_val_days: int,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """生成训练、验证以及推理所需的统一数据结构（低内存 + 可复用缓存版本）。"""
    if feature_set not in FEATURE_COLUMNS_MAP:
        raise ValueError(f'不支持的特征集合: {feature_set}')

    if cache_dir is None:
        from config import SETTINGS
        cache_dir = SETTINGS['paths']['cache_dir']

    feature_columns = FEATURE_COLUMNS_MAP[feature_set]
    cache_dir = _cache_dir_for_dataset(
        data_file=data_file,
        feature_set=feature_set,
        sequence_length=sequence_length,
        horizon=horizon,
        val_ratio=val_ratio,
        min_val_days=min_val_days,
        cache_base=cache_dir,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    sequence_file = cache_dir / 'sequences.npy'
    labels_file = cache_dir / 'labels.npy'
    train_scaled_file = cache_dir / 'train_sequences_scaled.npy'
    val_scaled_file = cache_dir / 'val_sequences_scaled.npy'
    meta_file = cache_dir / 'meta.csv'
    scaler_file = cache_dir / 'scaler.pkl'

    has_cache = all(
        path.exists()
        for path in [sequence_file, labels_file, train_scaled_file, val_scaled_file, meta_file, scaler_file]
    )

    if has_cache:
        meta_df = pd.read_csv(meta_file, dtype={'stock_id': str})
        meta_df['日期'] = pd.to_datetime(meta_df['日期'])
        meta_df['买入日'] = pd.to_datetime(meta_df['买入日'])
        meta_df['卖出日'] = pd.to_datetime(meta_df['卖出日'])
        scaler = joblib.load(scaler_file)
    else:
        raw_df = load_market_data(data_file)
        feature_df, _ = build_feature_frame(raw_df, feature_set=feature_set, num_processes=num_processes)
        sequence_file, labels_file, meta_df = _build_supervised_memmap(
            feature_df,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            horizon=horizon,
            output_dir=cache_dir,
            storage_dtype=np.float32,
        )

        train_mask, val_mask, _ = split_train_val_by_date(meta_df, val_ratio=val_ratio, min_val_days=min_val_days)
        train_indices = np.where(train_mask)[0].astype(np.int64)
        val_indices = np.where(val_mask)[0].astype(np.int64)

        scaler = _fit_scaler_from_memmap(
            sequence_file,
            train_indices=train_indices,
            feature_dim=len(feature_columns),
        )
        _transform_subset_to_memmap(
            sequence_file,
            scaler=scaler,
            subset_indices=train_indices,
            output_path=train_scaled_file,
            sequence_length=sequence_length,
            feature_dim=len(feature_columns),
            output_dtype=np.float32,
        )
        _transform_subset_to_memmap(
            sequence_file,
            scaler=scaler,
            subset_indices=val_indices,
            output_path=val_scaled_file,
            sequence_length=sequence_length,
            feature_dim=len(feature_columns),
            output_dtype=np.float32,
        )

        meta_df.to_csv(meta_file, index=False, encoding='utf-8-sig')
        joblib.dump(scaler, scaler_file)

        del raw_df
        del feature_df
        del train_mask
        del val_mask
        del train_indices
        del val_indices
        gc.collect()

    train_mask, val_mask, split_date = split_train_val_by_date(meta_df, val_ratio=val_ratio, min_val_days=min_val_days)
    train_indices = np.where(train_mask)[0].astype(np.int64)
    val_indices = np.where(val_mask)[0].astype(np.int64)

    labels_mm = np.load(labels_file, mmap_mode='r')
    train_labels = np.asarray(labels_mm[train_indices], dtype=np.float32)
    val_labels = np.asarray(labels_mm[val_indices], dtype=np.float32)

    train_sequences_scaled = np.load(train_scaled_file, mmap_mode='r+')
    val_sequences_scaled = np.load(val_scaled_file, mmap_mode='r+')
    train_meta = meta_df.loc[train_mask].reset_index(drop=True)
    val_meta = meta_df.loc[val_mask].reset_index(drop=True)

    del labels_mm
    del meta_df
    del train_mask
    del val_mask
    del train_indices
    del val_indices
    gc.collect()

    return {
        'feature_columns': feature_columns,
        'split_date': str(pd.Timestamp(split_date).date()),
        'train_sequences': train_sequences_scaled,
        'val_sequences': val_sequences_scaled,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'train_meta': train_meta,
        'val_meta': val_meta,
        'scaler': scaler,
    }


def build_inference_sequences(
    data_file: str | Path,
    sequence_length: int,
    feature_set: str,
    scaler: StandardScaler,
    num_processes: int = 0,
    cutoff_date: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """为最新可预测日期构建每只股票的序列输入。"""
    raw_df = load_market_data(data_file)
    if cutoff_date is not None:
        ts = pd.to_datetime(cutoff_date)
        raw_df = raw_df[raw_df['日期'] <= ts].copy()

    feature_df, feature_columns = build_feature_frame(raw_df, feature_set=feature_set, num_processes=num_processes)
    feature_df = feature_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    latest_date = feature_df['日期'].max()

    sequences: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for stock_id, group in feature_df.groupby('股票代码', sort=False):
        group = group[group['日期'] <= latest_date].sort_values('日期').tail(sequence_length)
        if len(group) != sequence_length:
            continue
        sequences.append(group[feature_columns].to_numpy(dtype=np.float32))
        meta_rows.append({'stock_id': stock_id, '日期': latest_date})

    if not sequences:
        raise ValueError('没有可用于推理的完整序列')

    seq_array = np.asarray(sequences, dtype=np.float32)
    scaled = transform_sequences(seq_array, scaler)
    meta_df = pd.DataFrame(meta_rows)
    meta_df['日期'] = pd.to_datetime(meta_df['日期'])
    return scaled, meta_df, feature_columns


def build_prediction_frame(
    meta_df: pd.DataFrame,
    scores: np.ndarray,
    top_k: int = 5,
    total_weight: float = 1.0,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """将模型分数转成比赛要求的 stock_id + weight 输出。"""
    if len(meta_df) != len(scores):
        raise ValueError('股票元信息数量与分数数量不一致')
    if total_weight <= 0 or total_weight > 1.0:
        raise ValueError('total_weight 必须在 (0, 1] 区间内')

    output = meta_df.copy()
    output['score'] = scores.astype(float)
    output = output.sort_values('score', ascending=False).head(top_k).reset_index(drop=True)
    logits = output['score'].to_numpy(dtype=np.float64)
    logits = (logits - logits.max()) / max(temperature, 1e-6)
    probs = np.exp(logits)
    weights = probs / probs.sum()
    output['weight'] = weights * total_weight
    return output.rename(columns={'stock_id': 'stock_id'})[['stock_id', 'weight']]


def save_bundle_metadata(output_file: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_scaler(output_file: str | Path, scaler: StandardScaler) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)


def load_scaler(input_file: str | Path) -> StandardScaler:
    return joblib.load(input_file)
