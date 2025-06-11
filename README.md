# DL2025_Team2_StockPrediction
Introduction to Deep Learning Course 2025 of CCU Team Project

## Model introduction

A GRU based stock price prediction model with GRU layers.
Implemented with Pytorch and trained on [stocknet-dataset](https://github.com/yumoxu/stocknet-dataset/)

## 如何使用

本專案透過 `uv` 進行虛擬環境管理，請先安裝 `uv`，
接著創立虛擬環境：

```bash
uv create venv
```

然後啟動虛擬環境：

```bash
source .venv/bin/activate
```

接著安裝依賴：

```bash
uv pip install -r requirements.txt
```

訓練模型：

```bash
python3 src/trainer.py --model GRU
```

在測試集上評估模型：

```bash
python3 src/inference.py --model GRU
```

## 如何使用

本專案透過 `uv` 進行虛擬環境管理，請先安裝 `uv`，
接著創立虛擬環境：

```bash
uv create venv
```

然後啟動虛擬環境：

```bash
source .venv/bin/activate
```

接著安裝依賴：

```bash
uv pip install -r requirements.txt
```

訓練模型：

```bash
python3 src/trainer.py --model GRU
```

在測試集上評估模型：

```bash
python3 src/inference.py --model GRU
```

## Dataset

[Stock Movement Prediction from Tweets and Historical Prices](https://aclanthology.org/P18-1183/) (Xu & Cohen, ACL 2018)