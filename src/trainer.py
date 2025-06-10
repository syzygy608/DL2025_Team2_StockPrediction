import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import CNNLSTMPredictor
from dataloader import TimeSeriesDataset
from evaluate import compute_accuracy


# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_loader, criterion, optimizer):
    """單個 epoch 的訓練邏輯"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    progress_bar = tqdm.tqdm(train_loader, desc="Training", unit="batch")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # 確保輸出形狀為 [batch_size]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_acc += compute_accuracy(outputs, targets) * batch_size
        total_samples += batch_size
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total_samples, running_acc / total_samples

def validate_epoch(model, val_loader, criterion):
    """單個 epoch 的驗證邏輯"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_acc += compute_accuracy(outputs, targets) * batch_size
            total_samples += batch_size

    return running_loss / total_samples, running_acc / total_samples

def train_model(batch_size, num_epochs, learning_rate, weight_decay):
    # 創建模型保存目錄
    os.makedirs("model_weights", exist_ok=True)

    # 載入數據
    train_dataset = load_dataset('train')
    val_dataset = load_dataset('val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、損失函數和優化器
    model = CNNLSTMPredictor().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 初始化 TensorBoard
    writer = SummaryWriter()

    # 最佳模型跟踪
    best_val_loss = float('inf')
    best_model_path = os.path.join("model_weights", "best_model.pth")

    # 訓練迴圈
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        # 打印結果
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 記錄到 TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with val loss: {best_val_loss:.4f}")

    writer.close()
    print("Training completed.")

def get_args():
    parser = argparse.ArgumentParser(description='GRU Predictor Training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_model(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )