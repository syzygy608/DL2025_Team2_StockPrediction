import torch
def binary_accuracy(outputs, targets):
    preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
    correct = (preds == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()
