import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class TimeSeriesDataset:
    def __init__(self, data, look_back=20):
        self.data = data
        self.look_back = look_back
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def min_max_normalization_tensor(self, tensor):
        copy_tensor = tensor.clone()
        min_val = copy_tensor.min(dim=0, keepdim=True).values
        max_val = copy_tensor.max(dim=0, keepdim=True).values
        range_val = torch.where(max_val - min_val == 0, torch.tensor(1.0, device=range_val.device), max_val - min_val)
        normalized_tensor = (copy_tensor - min_val) / range_val
        return normalized_tensor

    def generate_sequences(self, group, embedding_dim=10):
        """為單個公司生成時間序列片段"""
        # Convert date to time index
        group['Date'] = pd.to_datetime(group['Date'])
        group['Date'] = (group['Date'] - group['Date'].min()).dt.total_seconds() / (24 * 3600)

        # Prepare input features
        company_embeds_np = np.array(group['Company Embedding'].tolist())  # Shape: [n, 10]
        company_embeds = torch.tensor(company_embeds_np, dtype=torch.float32).to(self.device)
        other_features = torch.tensor(group[['Date', 'Open', 'Close', 'Adj Close', 'High', 'Low', 'Volume']].values, 
                                     dtype=torch.float32).to(self.device)  # Shape: [n, 7]
        inputs_tensor = torch.cat((company_embeds, other_features), dim=1)  # Shape: [n, 17]

        # Generate labels
        outputs = []
        for i in range(len(group)):
            if i == 0:
                outputs.append(0)
            else:
                outputs.append(1 if group['Adj Close'].iloc[i] > group['Adj Close'].iloc[i - 1] else 0)
        
        outputs_tensor = torch.tensor(outputs, dtype=torch.float).to(self.device)

        # Create time-series sequences
        tensors = []
        targets = []
        for i in range(len(inputs_tensor) - self.look_back):
            x = inputs_tensor[i:i + self.look_back]  # Shape: [look_back, 17]
            y = outputs_tensor[i + self.look_back - 1]  # Shape: []
            tensors.append(x)
            targets.append(y)
        
        return tensors, targets

    def create_dataset(self):
        """生成所有公司的序列和嵌入"""
        # Create vocabulary and embeddings
        vocab = {name: idx for idx, name in enumerate(self.data['Company Name'].unique())}
        vocab_size = len(vocab)
        embedding_dim = 10
        company_indices = [vocab[name] for name in self.data['Company Name'].values]
        company_indices_tensor = torch.tensor(company_indices, dtype=torch.long).to(self.device)
        embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(self.device)
        
        # Store embeddings
        embedded_company_names = embedding_layer(company_indices_tensor).cpu().detach().numpy()
        self.data['Company Embedding'] = list(embedded_company_names)
        
        return vocab  # Return vocab for company grouping

    def split_data(self, train_size=0.8, val_size=0.1, random_state=42):
        """按公司分割數據"""
        # Generate embeddings and get vocabulary
        _ = self.create_dataset()
        companies = list(self.data['Company Name'].unique())
        
        # Split companies into train, val, test
        train_companies, temp_companies = train_test_split(companies, train_size=train_size, random_state=random_state)
        val_ratio = val_size / (1 - train_size)  # Adjust for remaining data
        val_companies, test_companies = train_test_split(temp_companies, train_size=val_ratio, random_state=random_state)

        # Initialize lists for sequences
        train_tensors, train_targets = [], []
        val_tensors, val_targets = [], []
        test_tensors, test_targets = [], []

        # Process each company
        for name in self.data['Company Name'].unique():
            group = self.data[self.data['Company Name'] == name].sort_values(['Date'])
            if len(group) < self.look_back:
                print(f"Skipping {name}: only {len(group)} rows, need at least {self.look_back}")
                continue
            
            tensors, targets = self.generate_sequences(group)
            
            if name in train_companies:
                train_tensors.extend(tensors)
                train_targets.extend(targets)
            elif name in val_companies:
                val_tensors.extend(tensors)
                val_targets.extend(targets)
            elif name in test_companies:
                test_tensors.extend(tensors)
                test_targets.extend(targets)

        # Convert to tensors
        train_X = torch.stack(train_tensors) if train_tensors else torch.empty(0, self.look_back, 17, device=self.device)
        train_y = torch.stack(train_targets) if train_targets else torch.empty(0, device=self.device)
        val_X = torch.stack(val_tensors) if val_tensors else torch.empty(0, self.look_back, 17, device=self.device)
        val_y = torch.stack(val_targets) if val_targets else torch.empty(0, device=self.device)
        test_X = torch.stack(test_tensors) if test_tensors else torch.empty(0, self.look_back, 17, device=self.device)
        test_y = torch.stack(test_targets) if test_targets else torch.empty(0, device=self.device)

        # Normalize numerical features (indices 11 and beyond)
        if len(train_X) > 0:
            train_X[:, :, 11:] = self.min_max_normalization_tensor(train_X[:, :, 11:])
        if len(val_X) > 0:
            val_X[:, :, 11:] = self.min_max_normalization_tensor(val_X[:, :, 11:])
        if len(test_X) > 0:
            test_X[:, :, 11:] = self.min_max_normalization_tensor(test_X[:, :, 11:])

        self.train_data = (train_X, train_y)
        self.val_data = (val_X, val_y)
        self.test_data = (test_X, test_y)

    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def __len__(self):
        return len(self.data)