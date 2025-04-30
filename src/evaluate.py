import torch
import torch.nn as nn

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, y_pred, y_true):
        # Apply log transformation and calculate MSE
        y_pred = torch.log1p(y_pred)
        y_true = torch.log1p(y_true)
        return torch.sqrt(self.mse(y_pred, y_true))
    
def relative_error_accuracy(y_pred, y_true, eps=1e-8):
    """
    計算相對誤差準確度
    
    Args:
        y_pred: 預測值 (batch_size, 1)
        y_true: 真實值 (batch_size, 1)
        eps: 避免除零的小值
    
    Returns:
        accuracy: 相對誤差
    """
    relative_error = torch.abs(y_pred - y_true) / (y_true + eps)
    mre = relative_error.mean()
    accuracy = 1 - mre
    return accuracy