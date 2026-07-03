from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from config import SETTINGS
    from evaluation import score_prediction_file

    score = score_prediction_file(
        prediction_file=SETTINGS['paths']['output_file'],
        test_data_file=SETTINGS['paths']['test_data'],
        save_file=SETTINGS['paths']['score_file'],
    )
    print(f'LSTM 本地收益率得分: {score:.8f}')


if __name__ == '__main__':
    main()
