import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import CNNLSTMPredictor
from dataloader import load_dataset
from evaluate import compute_accuracy

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
            running_acc += compute_accuracy(outputs, targets) * batch_size
            total_samples += batch_size
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / total_samples
    avg_acc = running_acc / total_samples
    return avg_loss, avg_acc

def load_model(model_path):
    """載入模型並設置為評估模式"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    model = CNNLSTMPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def main():
    # 解析參數
    parser = argparse.ArgumentParser(description='GRU Predictor Evaluation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    args = parser.parse_args()

    # 載入測試數據
    test_dataset = load_dataset('test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 載入模型
    model_path = os.path.join("model_weights", "best_model.pth")
    model = load_model(model_path)

    # 定義損失函數
    criterion = nn.BCEWithLogitsLoss()

    # 初始化 TensorBoard
    writer = SummaryWriter()

    # 評估模型
    avg_loss, avg_acc = evaluate_model(model, test_loader, criterion)

    # 記錄到 TensorBoard
    writer.add_scalar('Test/Loss', avg_loss, 0)
    writer.add_scalar('Test/Accuracy', avg_acc, 0)

    # 打印結果
    print(f"Test Loss (BCE): {avg_loss:.4f}")
    print(f"Test Accuracy: {avg_acc:.4f}")

    writer.close()

if __name__ == "__main__":
    main()