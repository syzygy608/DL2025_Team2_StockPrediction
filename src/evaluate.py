def compute_accuracy(outputs, targets):
    outputs = outputs.sigmoid()  # Apply sigmoid to outputs
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    TP = ((outputs >= 0.5) & (targets == 1)).sum().item()
    TN = ((outputs < 0.5) & (targets == 0)).sum().item()
    FP = ((outputs >= 0.5) & (targets == 0)).sum().item()
    FN = ((outputs < 0.5) & (targets == 1)).sum().item()
    total = TP + TN + FP + FN
    if total == 0:
        return 0.0  # Avoid division by zero
    
    return (TP + TN) / total