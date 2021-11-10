import numpy as np
import torch.utils.data as data
import torch



class DatasetTrain(data.Dataset):
    def __init__(self, data_file, label_file, transform=None):
        self.features_all= np.load(data_file)[0: 145000]
        self.labels = np.load(label_file)[0: 145000]
    def __getitem__(self, index):
        features = self.features_all[index]
        target = self.labels[index]

        return torch.tensor(features).float(), torch.tensor(target).float()
    
    def __len__(self):
        return len(self.features_all)



class DatasetVal(data.Dataset):
    def __init__(self, data_file, label_file, transform=None):
        self.features_all= np.load(data_file)[14500: ]
        self.labels = np.load(label_file)[14500: ]
    def __getitem__(self, index):
        features = self.features_all[index]
        target = self.labels[index]

        return torch.tensor(features).float(), torch.tensor(target).float()
    
    def __len__(self):
        return len(self.features_all)

class DatasetTest(data.Dataset):
    def __init__(self, data_file, label_file, transform=None):
        self.features_all= np.load(data_file)
        self.labels = np.load(label_file)
    def __getitem__(self, index):
        features = self.features_all[index]
        target = self.labels[index]

        return torch.tensor(features).float(), torch.tensor(target).float()
    
    def __len__(self):
        return len(self.features_all)


