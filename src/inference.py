from model.model import GRUPredictor
from dataloader import load_dataset
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from evaluatealuate import RMSLELoss, relative_error_accuracy
import tqdm

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

    progress_bar = tqdm.tqdm(total=len(test_loader), desc="Testing", unit="batch")
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Get the model predictions
        outputs = model(inputs)

        # Calculate RMSLE
        criterion = RMSLELoss()
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        acc.append(relative_error_accuracy(outputs, targets).item())
        progress_bar.update(1)
    
    print(f"Average RMSLE on test set: {np.mean(losses) :.4f}")
    print(f"Average relative error accuracy on test set: {np.mean(acc) :.4f}")
    progress_bar.close()

if __name__ == "__main__":
    main()