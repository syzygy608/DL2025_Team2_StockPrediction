import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import TensorDataset

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

        print(f"Train companies: {len(train_companies)}")
        print(f"Val companies: {len(val_companies)}")
        print(f"Test companies: {len(test_companies)}")

    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def __len__(self):
        return len(self.data)
    
def download_stocknet_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "../dataset")
    url = "https://github.com/yumoxu/stocknet-dataset/archive/refs/heads/master.zip"  # StockNet GitHub 倉庫
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    zip_path = os.path.join(save_dir, "stocknet.zip")
    if not os.path.exists(zip_path) and not os.path.exists(os.path.join(save_dir, "extracted")):
        print("正在下載 StockNet dataset...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"下載失敗，狀態碼: {response.status_code}")
        with open(zip_path, "wb") as f:
            # 使用 tqdm 進度條顯示下載進度
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm.tqdm(total=total_size)
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                progress_bar.update(len(data))
            progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            raise Exception("下載不完整！")
        else:
            print("下載完成！")
    
    # 解壓文件
    extract_path = os.path.join(save_dir, "extracted")
    if not os.path.exists(extract_path):
        print("正在解壓數據...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print("解壓完成！")

    # 清除不必要的文件
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("刪除壓縮文件完成！")
    
    if os.path.exists(os.path.join(extract_path, "stocknet-dataset-master")):
        # 將 price/raw 內的資料移出來後刪除 stocknet-dataset-master
        print("正在移動資料...")
        for root, dirs, files in os.walk(os.path.join(extract_path, "stocknet-dataset-master")):
            for file in files:
                if file.endswith(".csv"):
                    src = os.path.join(root, file)
                    dst = os.path.join(save_dir, file)
                    if not os.path.exists(dst):
                        os.rename(src, dst)
                    else:
                        print(f"文件 {dst} 已存在，跳過移動。")
        
        # 刪除目錄 ( 不管有沒有資料 )
        for root, dirs, files in os.walk(extract_path, topdown=False):
            for name in dirs:
                os.rmdir(os.path.join(root, name))
            for name in files:
                os.remove(os.path.join(root, name))
        
        # 刪除 stocknet-dataset-master 目錄
        os.rmdir(extract_path)
        print("刪除 stocknet-dataset-master 目錄完成！")
    
    return save_dir

def preprocess_data(data_dir):
    # 數據預處理邏輯
    # 將所有 csv 檔案合併成一個 DataFrame，新增欄位 "Company Name" 來標識公司名稱
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            df["Company Name"] = file.split(".")[0]  # 使用檔名作為公司名稱
            all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.dropna()  # 去除缺失值
    combined_data = combined_data.reset_index(drop=True)  # 重設索引

    combined_data["Date"] = pd.to_datetime(combined_data["Date"])

    # 儲存預處理後的數據
    preprocessed_path = os.path.join(data_dir, "preprocessed_data.csv")
    combined_data.to_csv(preprocessed_path, index=False)
    print(f"預處理後的數據已儲存至 {preprocessed_path}")
    
def load_dataset(mode, data_dir="../dataset"):
    """
    Load the dataset based on the mode (train, val, test).
    """
    path = os.path.dirname(__file__)
    path = os.path.join(path, data_dir)
    
    if not os.path.exists(os.path.join(path, "preprocessed_data.csv")):
        data_dir = download_stocknet_dataset()
        preprocess_data(data_dir)
    
    data = pd.read_csv(os.path.join(path, "preprocessed_data.csv"))
    dataset = TimeSeriesDataset(data)
    # 分割數據集
    dataset.split_data()
    
    if mode == 'train':
        X, y = dataset.train_data
    elif mode == 'val':
        X, y = dataset.val_data
    elif mode == 'test':
        X, y = dataset.test_data
    else:
        raise ValueError("Invalid mode. Choose from 'train', 'val', or 'test'.")
    return TensorDataset(X, y)