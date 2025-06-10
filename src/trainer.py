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

from model.model import Predictor
from dataloader import load_dataset
from evaluate import compute_accuracy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_loader, criterion, optimizer):
    """Single epoch training logic"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    progress_bar = tqdm.tqdm(train_loader, desc="Training", unit="batch", ncols=100, leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # Ensure output shape [batch_size]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_acc += compute_accuracy(outputs, targets) * batch_size
        total_samples += batch_size

    return running_loss / total_samples, running_acc / total_samples

def validate_epoch(model, val_loader, criterion):
    """Single epoch validation logic"""
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
    # Create model save directory
    os.makedirs("model_weights", exist_ok=True)

    # Load data
    train_dataset = load_dataset('train')
    val_dataset = load_dataset('val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = Predictor().to(device)
        
    # Initialize loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-4)

    # Initialize TensorBoard
    writer = SummaryWriter()

    # Best model tracking
    best_val_loss = float('inf')
    best_model_path = os.path.join("model_weights", "best_model.pth")

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        # Update learning rate
        scheduler.step(val_loss)

        # Print results
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Log to TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with val loss: {best_val_loss:.4f}")

    writer.close()
    print("Training completed.")

def get_args():
    parser = argparse.ArgumentParser(description='CNN-LSTM Predictor Training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for optimizer')
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