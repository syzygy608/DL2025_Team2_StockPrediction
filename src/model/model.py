import torch.nn as nn
import torch.nn.functional as F

class GRUPredictor(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)  # 雙向 GRU 輸出維度翻倍
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out