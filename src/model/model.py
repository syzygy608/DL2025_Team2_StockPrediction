# model/model.py
import torch.nn as nn

class CNNLSTMPredictor(nn.Module):
    def __init__(self, input_dim=17, conv_filters=64, kernel_size=3, lstm_hidden_dim=128, dropout=0.3):
        """
        CNN + LSTM 模型用於股票二元趨勢預測
        Args:
            input_dim (int): 輸入特徵維度 (embedding_dim + 7 = 10 + 7)
            conv_filters (int): 卷積層的濾波器數量
            kernel_size (int): 卷積核大小
            lstm_hidden_dim (int): LSTM 隱藏層維度
            dropout (float): Dropout 比率，防止過擬合
        """
        super(CNNLSTMPredictor, self).__init__()
        
        # CNN 層：1D 卷積提取局部特徵
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=conv_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size//2  # 保持序列長度
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # LSTM 層：建模時間序列依賴
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 全連接層：映射到二元分類輸出
        self.fc = nn.Linear(lstm_hidden_dim, 1)
    
    def forward(self, x):
        # 將輸入轉置為 [batch_size, input_dim, look_back] 以適應 Conv1d
        x = x.permute(0, 2, 1)
        
        # CNN 層
        x = self.conv1(x)  # [batch_size, conv_filters, look_back]
        x = self.relu(x)
        x = self.dropout(x)
        
        # 轉置回 [batch_size, look_back, conv_filters] 以適應 LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM 層
        _, (hn, _) = self.lstm(x)  # hn: [num_layers, batch_size, lstm_hidden_dim]
        x = hn[-1]  # 取最後一層的隱藏狀態 [batch_size, lstm_hidden_dim]
        
        # 全連接層
        x = self.fc(x).squeeze(-1)  # [batch_size]
        return x