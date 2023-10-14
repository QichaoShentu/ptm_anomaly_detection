import os
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class SMAPSegLoader(Dataset): # 25
    # XXX
    def __init__(self, root_path, win_size, step=1, flag="train", finetune=False):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(os.path.join(root_path, "SMAP/SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        train_data_len = (int)(len(data) * 0.75) 
        self.train = data[:train_data_len] # 6/10
        if finetune:
            self.train = self.train[:train_data_len//2]
        self.val = data[train_data_len:] # 2/10

        test_data = np.load(os.path.join(root_path, "SMAP/SMAP_test.npy"))
        test_data = test_data[:len(self.val)] # 2/10
        self.test = self.scaler.transform(test_data)

        test_labels = np.load(os.path.join(root_path, "SMAP/SMAP_test_label.npy"))
        self.test_labels = test_labels[:len(self.val)]

        # print("train:", self.train.shape)
        # print("valid:", self.val.shape)
        # print("test:", self.test.shape)
        
    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SMDSegLoader(Dataset): # 38
    # XXX
    def __init__(self, root_path, win_size, step=1, flag="train", finetune=False):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(os.path.join(root_path, "SMD/SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        train_data_len = (int)(len(data) * 0.75) # 6/10
        self.train = data[:train_data_len]
        if finetune:
            self.train = self.train[:train_data_len//2]
        self.val = data[train_data_len:] # 2/10

        test_data = np.load(os.path.join(root_path, "SMD/SMD_test.npy"))
        test_data = test_data[:len(self.val)] # 2/10
        self.test = self.scaler.transform(test_data)
        
        test_labels = np.load(os.path.join(root_path, "SMD/SMD_test_label.npy"))
        self.test_labels = test_labels[:len(self.val)]

        # print("train:", self.train.shape)
        # print("valid:", self.val.shape)
        # print("test:", self.test.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
class GECCOSegLoader(Dataset): # 9
    # XXX
    def __init__(self, root_path, win_size, step=1, flag="train", finetune=False):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(os.path.join(root_path, "NIPS_TS_GECCO/NIPS_TS_Water_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        train_data_len = (int)(len(data) * 0.75) # 6/10
        self.train = data[:train_data_len]
        if finetune:
            self.train = self.train[:train_data_len//2]
        self.val = data[train_data_len:] # 2/10

        test_data = np.load(os.path.join(root_path, "NIPS_TS_GECCO/NIPS_TS_Water_test.npy"))
        test_data = test_data[:len(self.val)] # 2/10
        self.test = self.scaler.transform(test_data)
        
        test_labels = np.load(os.path.join(root_path, "NIPS_TS_GECCO/NIPS_TS_Water_test_label.npy"))
        self.test_labels = test_labels[:len(self.val)]

        # print("train:", self.train.shape)
        # print("valid:", self.val.shape)
        # print("test:", self.test.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.zeros(self.win_size, dtype=np.float32)
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
 
if __name__ == '__main__':
    # SMAP
    # dataset = SMAPSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='train')
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # dataset = SMAPSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='val')
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # dataset = SMAPSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='test')
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # SMD
    # dataset = SMDSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='train')
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # dataset = SMDSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='val')
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # dataset = SMDSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='test')
    # print(len(dataset))
    # data, labels = dataset.__getitem__(0)
    # print(data.shape)
    # print(labels.shape)

    # GECCO
    dataset = GECCOSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='train')
    print(len(dataset))
    data, labels = dataset.__getitem__(0)
    print(data.shape)
    print(labels.shape)

    dataset = GECCOSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='val')
    print(len(dataset))
    data, labels = dataset.__getitem__(0)
    print(data.shape)
    print(labels.shape)

    dataset = GECCOSegLoader('/workspace/ptm_anomaly_detection/dataset', 100, flag='test')
    print(len(dataset))
    data, labels = dataset.__getitem__(0)
    print(data.shape)
    print(labels.shape)