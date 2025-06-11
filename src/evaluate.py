import torch
def binary_accuracy(outputs, targets):
    """
    計算二元分類的準確率
    Args:
        outputs (torch.Tensor): 模型輸出，形狀為 [batch_size, 1]
        targets (torch.Tensor): 真實標籤，形狀為 [batch_size, 1]
    Returns:
        float: 準確率
    """
    preds = (outputs > 0.5).float()  # 將輸出轉換為二元標籤
    correct = (preds == targets).float()  # 計算正確預測
    accuracy = correct.sum() / correct.numel()  # 計算準確率
    return accuracy.item()  # 返回準確率值
