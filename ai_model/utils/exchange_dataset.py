import torch
from torch.utils.data import Dataset

class ExchangeRateDataset(Dataset):
    def __init__(self, X, y, gap=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.gap = None if gap is None else torch.tensor(gap, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.gap is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.gap[idx]