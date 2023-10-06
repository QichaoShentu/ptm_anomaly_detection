import os
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class PSMSegLoader(Dataset): # dim=25
    # OK 
    def __init__(self, root_path, win_size, step=1):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'PSM/train.csv'))
        train_data = train_data.values[:, 1:]
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data

        # print("train:", self.train.shape)

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index:index+self.win_size]), np.zeros(self.win_size, dtype=np.float32)


class MSLSegLoader(Dataset): # dim=55
    def __init__(self, root_path, win_size, step=1):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # XXX
        train_data = np.load(os.path.join(root_path, "MSL/MSL_train.npy"))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data
        
        # print("train:", self.train.shape)

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1
       
    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index:index+self.win_size]), np.zeros(self.win_size, dtype=np.float32)


class SWATSegLoader(Dataset): # dim=51
    # OK
    def __init__(self, root_path, win_size, step=1):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'SWaT/swat_train2.csv'))
        train_data = train_data.values[:, :-1]
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data
        
        # print("train:", self.train.shape)

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
    

class ASDSegLoader(Dataset): # machime-* dim=38, omi-* dim=19
    def __init__(self, root_path, win_size, step=1):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        # XXX
        with open(os.path.join(root_path, 'ASD/machine-1-1_train.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data
        
        # print("train:", self.train.shape)

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
    

class SKABSegLoader(Dataset): # dim=8
    # OK
    def __init__(self, root_path, win_size, step=1):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'SKAB/anomaly-free/anomaly-free.csv'), index_col='datetime', sep=';')
        train_data = train_data.values
        # print(train_data.shape)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data
        
        # print("train:", self.train.shape)

    def __len__(self):
        return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)


if __name__ == '__main__':
    # PSM
    # dataset = PSMSegLoader('/workspace/ptm_anomaly_detection/dataset', 100)
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # MSL
    # dataset = MSLSegLoader('/workspace/ptm_anomaly_detection/dataset', 100)
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # SWaT
    # dataset = SWATSegLoader('/workspace/ptm_anomaly_detection/dataset', 100)
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # ASD
    dataset = ASDSegLoader('/workspace/ptm_anomaly_detection/dataset', 100)
    print(len(dataset))
    data, labels = dataset.__getitem__(0)
    print(data.shape)
    print(labels.shape)

    # SKAB
    # dataset = SKABSegLoader('/workspace/ptm_anomaly_detection/dataset', 100)
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)