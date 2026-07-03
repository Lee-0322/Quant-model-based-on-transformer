import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# code/src的父目录是code，code的父目录是项目根目录
PROJECT_ROOT = BASE_DIR.parent.parent
# APP_DIR用于Docker环境，默认使用项目根目录
APP_DIR = os.environ.get('APP_DIR', PROJECT_ROOT)

SETTINGS = {
    'feature_set': '158+39',
    'sequence_length': 60,
    'score_horizon': 5,
    'num_processes': 0,
    'val_ratio': 0.15,
    'min_val_days': 20,
    'top_k': 5,
    'total_weight': 1.0,
    'temperature': 0.7,
    'seed': 42,
    'model': {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
    },
    'train': {
        'batch_size': 256,
        'epochs': 20,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'patience': 4,
    },
    'paths': {
        'train_data': Path(APP_DIR) / 'data' / 'train.csv',
        'test_data': Path(APP_DIR) / 'data' / 'test.csv',
        'model_file': Path(APP_DIR) / 'model' / 'lstm_model.pth',
        'scaler_file': Path(APP_DIR) / 'model' / 'sequence_scaler.pkl',
        'metadata_file': Path(APP_DIR) / 'model' / 'train_metadata.json',
        'output_file': Path(APP_DIR) / 'output' / 'result.csv',
        'score_file': Path(APP_DIR) / 'output' / 'local_score.csv',
        'cache_dir': Path(APP_DIR) / 'temp' / 'cache',
    },
}
