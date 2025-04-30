import os
import requests
import zipfile
import tqdm
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# 1. 自動下載數據集
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
    # 將所有 csv 檔案合併成一個 DataFrame，新增欄位 company_id，以第幾個檔案為 id
    all_data = []
    idx = 1
    for i, file in enumerate(os.listdir(data_dir)):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            df["Company_id"] = idx  # 新增欄位 company_id
            all_data.append(df)
            idx += 1
    
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.dropna()  # 去除缺失值
    combined_data = combined_data.reset_index(drop=True)  # 重設索引

    # 將 Date 切分成 年、月、日三個欄位
    combined_data["Date"] = pd.to_datetime(combined_data["Date"])
    combined_data["Year"] = combined_data["Date"].dt.year
    combined_data["Month"] = combined_data["Date"].dt.month
    combined_data["Day"] = combined_data["Date"].dt.day
    combined_data = combined_data.drop(columns=["Date"])  # 刪除原始的 Date 欄位
    
    # 將 Company_id 放到最前面
    cols = list(combined_data.columns)
    cols.insert(0, cols.pop(cols.index("Company_id")))
    combined_data = combined_data[cols]

    # 儲存預處理後的數據
    preprocessed_path = os.path.join(data_dir, "preprocessed_data.csv")
    combined_data.to_csv(preprocessed_path, index=False)
    print(f"預處理後的數據已儲存至 {preprocessed_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset:
    def __init__(self, data, look_back=7):
        self.data = data
        self.look_back = look_back
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def logp_tensor(self, tensor):
        # 複製輸入張量，避免修改原始數據
        logp_tensor = tensor.clone()
        # 對張量進行對數變換
        logp_tensor = torch.log1p(logp_tensor)
        return logp_tensor

    def create_dataset(self):
        input_cols = ['Company_id', 'Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        output_cols = ['Close']

        tensors = []
        targets = []

        for id in self.data['Company_id'].unique():
            group = self.data[self.data['Company_id'] == id].sort_values(['Year', 'Month', 'Day'])

            if len(group) < self.look_back:
                print(f"Skipping {id}: only {len(group)} rows, need at least {self.look_back}")
                continue
        
            inputs = group[input_cols].values
            outputs = group[output_cols].values
            
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
            outputs_tensor = torch.tensor(outputs, dtype=torch.float32).to(device)

            for i in range(len(inputs_tensor) - self.look_back):
                x = inputs_tensor[i:i + self.look_back]
                y = outputs_tensor[i + self.look_back - 1]
                tensors.append(x)
                targets.append(y)
            
        tensors = torch.stack(tensors)
        targets = torch.stack(targets)
        return tensors, targets

    def split_data(self, train_size=0.8, val_size=0.1):
        X, y = self.create_dataset()
        total_len = len(X)
        train_len = int(total_len * train_size)
        val_len = int(total_len * val_size)

        train_X, val_X, test_X = X[:train_len], X[train_len:train_len + val_len], X[train_len + val_len:]
        train_y, val_y, test_y = y[:train_len], y[train_len:train_len + val_len], y[train_len + val_len:]

        train_X[:, :, 2:] = self.logp_tensor(train_X[:, :, 2:])
        val_X[:, :, 2:] = self.logp_tensor(val_X[:, :, 2:])
        test_X[:, :, 2:] = self.logp_tensor(test_X[:, :, 2:])

        train_y = self.logp_tensor(train_y)
        val_y = self.logp_tensor(val_y)
        test_y = self.logp_tensor(test_y)

        self.train_data = (train_X, train_y)
        self.val_data = (val_X, val_y)
        self.test_data = (test_X, test_y)
    
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def __len__(self):
        return len(self.data)

def load_dataset(mode, data_dir="../dataset"):
    """
    Load the dataset based on the mode (train, val, test).
    """
    path = os.path.dirname(__file__)
    path = os.path.join(path, data_dir)
    data = pd.read_csv(os.path.join(path, 'preprocessed_data.csv'))
    dataset = TimeSeriesDataset(data=data, look_back=7)
    dataset.split_data(train_size=0.8, val_size=0.1)
    
    if mode == 'train':
        X, y = dataset.train_data
    elif mode == 'val':
        X, y = dataset.val_data
    elif mode == 'test':
        X, y = dataset.test_data
    else:
        raise ValueError("Invalid mode. Choose from 'train', 'val', or 'test'.")
    return TensorDataset(X, y)

def print_dataset(dataset, max_samples=10):
    """Print the contents of the dataset, up to max_samples."""
    print(f"Dataset size: {len(dataset)}")
    print("Sample format: (inputs, targets)")
    
    for i, (inputs, targets) in enumerate(dataset):
        if i >= max_samples:
            print(f"... (showing only first {max_samples} samples)")
            break
        print(f"Sample {i}:")
        print(f"  Inputs shape: {inputs.shape}, Inputs: {inputs}")
        print(f"  Targets shape: {targets.shape}, Targets: {targets}")
        print()

# 3. 主函數：下載並構建 DataSet
def main():
    # 下載數據
    if os.path.exists(os.path.join("../dataset", "preprocessed_data.csv")):
        print("數據已存在，跳過下載。")
    else:
        data_dir = download_stocknet_dataset()
        # 數據預處理
        preprocess_data(data_dir)
    print("數據下載和預處理完成！")

        # Load and preprocess data
    train_dataset = load_dataset('train')
    val_dataset = load_dataset('val')
    test_dataset = load_dataset('test')
    
    print_dataset(train_dataset)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

if __name__ == "__main__":
    main()