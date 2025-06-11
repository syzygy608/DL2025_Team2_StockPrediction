import torch
import torch.nn as nn

def compute_accuracy(outputs, targets):
    """Matthews Correlation Coefficient"""
    outputs = torch.sigmoid(outputs)  # Apply sigmoid to outputs
    predictions = (outputs > 0.5).float()  # Convert to binary predictions
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator