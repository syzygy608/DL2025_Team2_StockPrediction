import torch
import torch.nn as nn

def compute_accuracy(outputs, targets):
    """計算二元分類的準確率"""
    preds = (torch.sigmoid(outputs) > 0.5).float()  # 將 logits 轉為 0/1 預測
    correct = (preds == targets).float().sum()
    return correct / targets.size(0)