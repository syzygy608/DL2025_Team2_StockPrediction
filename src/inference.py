import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import Predictor, GRUPredictor
from dataloader import load_dataset
from evaluate import directional_accuracy 

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, criterion):
    """評估模型在測試集上的性能"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    progress_bar = tqdm.tqdm(test_loader, desc="Testing", unit="batch")
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs).squeeze()  # 確保輸出形狀為 [batch_size]
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_acc += directional_accuracy(outputs, targets) * batch_size
            total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    return avg_loss, avg_acc

def main():
    # 解析參數
    parser = argparse.ArgumentParser(description='GRU Predictor Evaluation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--model', type=str, default='CNNLSTM', choices=['CNNLSTM', 'GRU'], help='Model type to use for evaluation')
    args = parser.parse_args()

    # 載入測試數據
    test_dataset = load_dataset('test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'CNNLSTM':
        model_path = '../model_weights/CCNLSTM_best_model.pth'
        model = Predictor()
    elif args.model == 'GRU':
        model_path = '../model_weights/GRU_best_model.pth'
        model = GRUPredictor()
    else:
        raise ValueError("Unsupported model type. Choose 'CNNLSTM' or 'GRU'.")
    
    # 載入模型
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model from {model_path}")

    # 定義損失函數
    criterion = nn.HuberLoss()

    # 初始化 TensorBoard
    writer = SummaryWriter()

    # 評估模型
    avg_loss, avg_acc = evaluate_model(model, test_loader, criterion)

    # 記錄到 TensorBoard
    writer.add_scalar('Test/Loss', avg_loss, 0)
    writer.add_scalar('Test/Accuracy', avg_acc, 0)

    print(f"Test Loss): {avg_loss:.4f}")
    print(f"Test Accuracy: {avg_acc:.4f}")

    writer.close()

if __name__ == "__main__":
    main()