import os
import requests
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 自動下載數據集
def download_stocknet_dataset(save_dir="../dataset"):
    url = "https://github.com/yumoxu/stocknet-dataset/archive/refs/heads/master.zip"  # StockNet GitHub 倉庫
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    zip_path = os.path.join(save_dir, "stocknet.zip")
    if not os.path.exists(zip_path):
        print("正在下載 StockNet dataset...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("下載完成！")
    
    # 解壓文件
    extract_path = os.path.join(save_dir, "extracted")
    if not os.path.exists(extract_path):
        print("正在解壓數據...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print("解壓完成！")
    
    return os.path.join(extract_path, "stocknet-dataset-master")

# 2. 自定義數據集類
class StockNetDataset(Dataset):
    def __init__(self, data_dir, window_size=5, transform=None):
        """
        Args:
            data_dir (str): 資料文件夾路徑
            window_size (int): 用於時間序列的窗口大小
            transform (callable, optional): 數據轉換函數
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.transform = transform
        
        # 加載價格數據（假設有一個 price.csv 文件）
        price_file = os.path.join(data_dir, "price", "raw", "price.csv")  # 根據實際文件路徑調整
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"未找到價格數據文件: {price_file}")
        
        self.price_data = pd.read_csv(price_file)
        self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
        self.price_data.set_index('Date', inplace=True)
        
    
    def __len__(self):
        return len(self.features) - self.window_size + 1
    
    def __getitem__(self, idx):
        # 獲取時間窗口數據
        window_data = self.features[idx:idx + self.window_size]
        
        # 轉換為 PyTorch 張量
        window_data = torch.tensor(window_data, dtype=torch.float32)
        
        if self.transform:
            window_data = self.transform(window_data)
        
        return window_data

# 3. 主函數：下載並構建 DataLoader
def main():
    # 下載數據
    data_dir = download_stocknet_dataset()
    
    # 初始化數據集
    dataset = StockNetDataset(data_dir=data_dir, window_size=5)
    
    # 創建 DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    for batch_features, batch_targets in dataloader:
        print(f"Batch Features Shape: {batch_features.shape}")  # [batch_size, window_size, num_features]

if __name__ == "__main__":
    main()