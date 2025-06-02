import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import GRUPredictor
from dataloader import load_dataset
from evaluate import RMSELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    # Load the model
    model = GRUPredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='Test the GRU model')
    parser.add_argument('--model_path', type=str, default="../model_weights/best_model.pth", help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    args = parser.parse_args()

    # Load the test dataset
    test_dataset = load_dataset('test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the model
    model = load_model(args.model_path)

    losses = []
    acc = []

    writer = SummaryWriter()

    criterion = nn.HuberLoss(delta=1.0)

    progress_bar = tqdm.tqdm(total=len(test_loader), desc="Testing", unit="batch")
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Get the model predictions
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        losses.append(loss.item())
        acc.append(RMSELoss()(outputs, targets).item())

        writer.add_scalar('Test/True', targets, batch_idx)
        writer.add_scalar('Test/Output', outputs, batch_idx)

        progress_bar.update(1)
    
    # Log average metrics
    avg_loss = np.mean(losses)
    avg_acc = np.mean(acc)

    print(f"Average Hubert Loss on test set: {avg_loss:.4f}")
    print(f"Average RMSE on test set: {avg_acc:.4f}")
    progress_bar.close()
    writer.close()

if __name__ == "__main__":
    main()