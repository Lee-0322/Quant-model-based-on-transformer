from __future__ import annotations

import sys
from pathlib import Path

import torch


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main() -> None:
    from config import SETTINGS
    from dataset import build_inference_sequences, build_prediction_frame, load_scaler
    from model import LSTMRegressor

    checkpoint = torch.load(SETTINGS['paths']['model_file'], map_location='cpu')
    scaler = load_scaler(SETTINGS['paths']['scaler_file'])
    device = resolve_device()

    model = LSTMRegressor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['model_config']['hidden_dim'],
        num_layers=checkpoint['model_config']['num_layers'],
        dropout=checkpoint['model_config']['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    sequences, meta_df, _ = build_inference_sequences(
        data_file=SETTINGS['paths']['train_data'],
        sequence_length=SETTINGS['sequence_length'],
        feature_set=SETTINGS['feature_set'],
        scaler=scaler,
        num_processes=SETTINGS['num_processes'],
    )

    with torch.no_grad():
        tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
        predictions = model(tensor).detach().cpu().numpy()

    result_df = build_prediction_frame(
        meta_df,
        predictions,
        top_k=SETTINGS['top_k'],
        total_weight=SETTINGS['total_weight'],
        temperature=SETTINGS['temperature'],
    )
    SETTINGS['paths']['output_file'].parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(SETTINGS['paths']['output_file'], index=False)
    print(result_df)


if __name__ == '__main__':
    main()
