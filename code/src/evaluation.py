from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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


def validate_prediction_df(prediction_df: pd.DataFrame) -> None:
    if 'stock_id' not in prediction_df.columns or 'weight' not in prediction_df.columns:
        raise ValueError('预测结果必须包含 stock_id 和 weight 两列')
    if len(prediction_df) > 5:
        raise ValueError('预测结果最多只能包含 5 支股票')
    weight_sum = float(prediction_df['weight'].sum())
    if not (0.0 <= weight_sum <= 1.0 + 1e-8):
        raise ValueError(f'预测结果权重之和必须位于 [0, 1]，当前为 {weight_sum}')


def score_prediction_file(
    prediction_file: str | Path,
    test_data_file: str | Path,
    save_file: str | Path | None = None,
) -> float:
    """对模型输出结果做本地加权收益率评分。"""
    prediction_df = pd.read_csv(prediction_file, dtype={'stock_id': str})
    prediction_df['stock_id'] = prediction_df['stock_id'].astype(str).str.zfill(6)
    validate_prediction_df(prediction_df)

    test_df = pd.read_csv(test_data_file, dtype={'股票代码': str})
    test_df['股票代码'] = test_df['股票代码'].astype(str).str.zfill(6)
    test_df['日期'] = pd.to_datetime(test_df['日期'])
    test_df = test_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    selected = test_df[test_df['股票代码'].isin(prediction_df['stock_id'])].copy()
    selected = selected.groupby('股票代码', group_keys=False).tail(5)
    if selected.empty:
        raise ValueError('测试集与预测股票没有交集，无法评分')

    rows: list[dict[str, float | str]] = []
    for stock_id, group in selected.groupby('股票代码'):
        group = group.sort_values('日期').reset_index(drop=True)
        if len(group) < 5:
            continue
        start_open = float(group.iloc[0]['开盘'])
        end_open = float(group.iloc[-1]['开盘'])
        stock_return = (end_open - start_open) / (start_open + 1e-12)
        weight = float(prediction_df.loc[prediction_df['stock_id'] == stock_id, 'weight'].iloc[0])
        rows.append({'stock_id': stock_id, 'return': stock_return, 'weight': weight})

    if not rows:
        raise ValueError('没有足够的 5 日测试窗口用于评分')

    result_df = pd.DataFrame(rows)
    final_score = float((result_df['return'] * result_df['weight']).sum())

    if save_file is not None:
        save_path = Path(save_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{'Final Score': final_score}]).to_csv(save_path, index=False)

    return final_score
