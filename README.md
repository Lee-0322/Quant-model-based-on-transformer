## 1. 项目目标与整体流程

核心目标是预测"未来5日收益率"并选出最优的Top5股票。

训练与推理主流程如下：
1. 读取历史行情数据（`data/train.csv`）；
2. 做特征工程（158+39特征）；
3. 构建标签：T+1买入、T+5卖出的收益率；
4. 按日期切分训练集/验证集，构建监督学习序列样本；
5. 训练LSTM模型，按验证集Top-K收益保存最优权重；
6. 使用训练好的模型在最新日期上生成Top5选股结果。

---

## 2. 代码结构说明

### [config.py](code/src/config.py)
统一管理训练与推理参数，包括：
- 序列长度 `sequence_length`（默认60）；
- 模型超参数（`hidden_dim`、`num_layers`、`dropout`）；
- 训练超参数（`batch_size`、`epochs`、`learning_rate`、`weight_decay`、`patience`）；
- 评分超参数（`top_k`、`temperature`、`total_weight`）；
- 数据路径和输出路径。

### [model.py](code/src/model.py)
定义核心模型 `LSTMRegressor`，主要由以下模块组成：
- `LSTMEncoder`：双层LSTM编码器，提取股票历史序列表示；
- `SequenceRegressionDataset`：PyTorch数据集类；
- `train_torch_model`：统一的PyTorch训练入口；
- `mean_daily_topk_return`：按日期Top-K计算平均收益。

输入形状：`[batch, seq_len, feature_dim]`
输出形状：`[batch]`，即每只股票的收益率预测值。

### [dataset.py](code/src/dataset.py)
包含数据处理与特征工程逻辑：
- `load_market_data()`：读取原始行情数据；
- `build_feature_frame()`：对全部股票做特征工程；
- `prepare_dataset_bundle()`：生成训练/验证数据结构（支持memmap缓存）；
- `build_inference_sequences()`：为推理构建序列输入；
- `build_prediction_frame()`：将模型分数转成股票权重输出。

特征工程使用了 `TA-Lib`，若未正确安装会报错。

### [utils.py](code/src/utils.py)
包含特征工程函数：
- `engineer_features_39()`：39个技术指标特征（SMA、EMA、RSI、MACD、KDJ等）；
- `engineer_features()`：158个Alpha类特征；
- `engineer_features_158plus39()`：合并158+39特征。

### [train.py](code/src/train.py)
训练主脚本，流程：
1. 设置随机种子保证可复现性；
2. 调用 `prepare_dataset_bundle` 准备数据；
3. 初始化 `LSTMRegressor` 模型；
4. 调用 `train_torch_model` 训练模型；
5. 保存模型权重、标准差器和元数据。

训练产物：
- `lstm_model.pth`：最佳模型参数；
- `sequence_scaler.pkl`：特征标准化器；
- `train_metadata.json`：训练元数据（包含验证集收益等）。

### [predict.py](code/src/predict.py)
推理主脚本，流程：
1. 加载模型权重和标准化器；
2. 调用 `build_inference_sequences` 构建推理序列；
3. 用模型预测每只股票的收益率；
4. 按预测分数取Top5，通过softmax转换为权重；
5. 输出到 `result.csv`（`stock_id` + `weight`）。

### [test.py](code/src/test.py)
本地评分脚本，根据预测结果计算加权收益率得分。

---

## 3. 数据与输入输出约定

### 输入数据
默认训练数据文件：
- `data/train.csv`（在Docker中挂载为 `/app/data/train.csv`）
- `data/test.csv`（在Docker中挂载为 `/app/data/test.csv`）

关键列：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅` 等。

### 预测输出文件
- `output/result.csv`（由 `predict.py` 生成）

输出格式：
```
stock_id,weight
000001,0.25
000002,0.25
000004,0.20
000005,0.15
000006,0.15
```

---



