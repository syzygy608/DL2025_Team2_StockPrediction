import torch
def directional_accuracy(y_true, y_pred, ignore_zero_diff=True):
    """
    計算 Directional Accuracy，支持忽略價格不變的情況。
    
    參數：
        y_true (torch.Tensor): 實際 Adjusted Close Price
        y_pred (torch.Tensor): 預測 Adjusted Close Price
        ignore_zero_diff (bool): 是否忽略價格不變的情況
    
    返回：
        float: Directional Accuracy 值
    """
    
    true_diff = y_true[1:] - y_true[:-1]
    pred_diff = y_pred[1:] - y_pred[:-1]
    
    true_sign = torch.sign(true_diff)
    pred_sign = torch.sign(pred_diff)
    
    if ignore_zero_diff:
        # 僅考慮實際價格有變動的情況（true_diff != 0）
        non_zero_mask = true_diff != 0
        true_sign = true_sign[non_zero_mask]
        pred_sign = pred_sign[non_zero_mask]
        
        if len(true_sign) == 0:  # 防止所有差值為 0
            return 0.0
    
    correct_directions = (true_sign == pred_sign).float()
    directional_acc = torch.mean(correct_directions)
    
    return directional_acc.item()
