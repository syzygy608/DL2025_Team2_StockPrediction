import torch.nn as nn
import torch.nn.functional as F

class GRUPredictor(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=1, dropout=0):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU 層
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # 全連接層
        self.fc = nn.Linear(hidden_size, output_size)
        # ReLU 激活函數
        self.relu = nn.ReLU()

    def forward(self, x):
        # GRU 前向傳播
        out, _ = self.gru(x)
        # 取最後一個時間步的輸出
        out = self.fc(out[:, -1, :])
        # 使用 ReLU 激活函數
        out = self.relu(out)
        return out