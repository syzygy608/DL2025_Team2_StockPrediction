from model.model import GRUPredictor
from dataloader import load_dataset
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluate import RMSLELoss, relative_error_accuracy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(batch_size, num_epochs, learning_rate, weight_decay):
    # get data
    train_dataset = load_dataset('train')
    val_dataset = load_dataset('val')
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function and optimizer
    model = GRUPredictor().to(device)
    # loss: rmsle
    criterion = RMSLELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Tensorboard
    writer = SummaryWriter()
    
    # best model
    best_model = None
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        acc = 0.0
        progress_bar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            acc += relative_error_accuracy(outputs, targets).item()
            progress_bar.update(1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_acc += relative_error_accuracy(outputs, targets).item()

        # Print statistics
        train_loss_avg = running_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc_avg = acc / len(train_loader)
        val_acc_avg = val_acc / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")
        print(f"Train Accuracy: {train_acc_avg:.4f}, Val Accuracy: {val_acc_avg:.4f}")
        # Log to Tensorboard
        writer.add_scalar('Loss/train', train_loss_avg, epoch)
        writer.add_scalar('Loss/val', val_loss_avg, epoch)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model = model.state_dict()
            print(f"Best model saved at epoch {epoch+1} with val loss: {best_val_loss:.4f}")
            current_dir = os.path.dirname(__file__)
            parrent_dir = os.path.join(current_dir, "../")
            torch.save(best_model, os.path.join(parrent_dir, '/saved_model/best_model.pth'))

    writer.close()
    progress_bar.close()

# 設定 hyperparameters : batch_size, num_epochs, learning_rate

def get_args():
    args = argparse.ArgumentParser(description='GRU Predictor Training')
    args.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    args.add_argument('--num_epochs', type=int, default=40, help='Number of epochs for training')
    args.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    args.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')

    return args.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_model(args.batch_size, args.num_epochs, args.learning_rate, args.weight_decay)
    print("Training completed.")