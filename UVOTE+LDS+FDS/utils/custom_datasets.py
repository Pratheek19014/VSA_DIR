import torch
from torch.utils.data import Dataset
import warnings

class sequential_dataset(Dataset):
    def __init__(
        self,
        df,
        target,
        seq_length,
        columns_to_drop,
        dtype=torch.float32,
    ):
        self.data = df.copy(deep=True)
        self.seq_length = seq_length

        self.y = torch.tensor(self.data[target].to_numpy(), dtype=dtype)
        self.X_columns = self.data.drop(columns_to_drop, axis=1).columns
        self.X = torch.tensor(
            self.data.drop(columns_to_drop, axis=1).to_numpy(),
            dtype=dtype,
        )
        self.densities = None

    def set_densities(self, densities):
        self.densities = torch.tensor(densities, dtype=torch.float32)

    def __getitem__(self, index):
        # Original sliding window logic
        x = self.X[index : index + self.seq_length]
        y = self.y[index + self.seq_length - 1]
        if self.densities is not None:
            density = self.densities[index + self.seq_length - 1]
            return x, y, density
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_length

class unsupervised_sequential_dataset(Dataset):
    def __init__(
        self,
        df,
        seq_length,
        columns_to_drop,
        dtype=torch.float32,
    ):
        self.data = df.copy(deep=True)
        self.seq_length = seq_length
        
        self.X = torch.tensor(
            self.data.drop(columns_to_drop, axis=1).to_numpy(),
            dtype=dtype,
        )

    def __getitem__(self, index):
        return self.X[index : index + self.seq_length]

    def __len__(self):
        return len(self.data) - self.seq_length