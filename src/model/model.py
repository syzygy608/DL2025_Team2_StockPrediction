import torch
import torch.nn as nn

class CNNLSTMPredictor(nn.Module):
    def __init__(self, input_dim=17, conv_filters=128, kernel_size=3, lstm_hidden_dim=256, dropout=0.3, num_layers=2):
        """
        Enhanced CNN + LSTM model for stock binary trend prediction
        Args:
            input_dim (int): Input feature dimension (embedding_dim + 7 = 10 + 7)
            conv_filters (int): Number of convolutional filters
            kernel_size (int): Convolutional kernel size
            lstm_hidden_dim (int): LSTM hidden dimension
            dropout (float): Dropout rate to prevent overfitting
            num_layers (int): Number of LSTM layers
        """
        super(CNNLSTMPredictor, self).__init__()
        
        # CNN layers: Two 1D convolutional layers for better feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.bn1 = nn.BatchNorm1d(conv_filters)  # Batch normalization
        self.conv2 = nn.Conv1d(
            in_channels=conv_filters,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.bn2 = nn.BatchNorm1d(conv_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # LSTM layer: Multi-layer for better sequence modeling
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers: Deeper output mapping
        self.fc1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2)
        self.fc2 = nn.Linear(lstm_hidden_dim // 2, 1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to improve training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Forward pass
        Args:
            x (torch.Tensor): Input shape [batch_size, look_back, input_dim]
        Returns:
            torch.Tensor: Output shape [batch_size], logits for binary classification
        """
        # Transpose for Conv1d: [batch_size, input_dim, look_back]
        x = x.permute(0, 2, 1)
        
        # CNN layers
        x = self.conv1(x)  # [batch_size, conv_filters, look_back]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)  # [batch_size, conv_filters, look_back]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Transpose for LSTM: [batch_size, look_back, conv_filters]
        x = x.permute(0, 2, 1)
        
        # LSTM layer
        _, (hn, _) = self.lstm(x)  # hn: [num_layers, batch_size, lstm_hidden_dim]
        x = hn[-1]  # Last layer's hidden state: [batch_size, lstm_hidden_dim]
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)  # [batch_size]
        return x