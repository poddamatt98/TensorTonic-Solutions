import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, data: list, label_col: int):
        self.data = data
        self.label_col = label_col

    def __len__(self) -> int:
        """
        Returns the number of rows.
        """
        return len(self.data)
        

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, label) as float32 tensors of shapes (D,) and (1,).
        """
        item = self.data[idx]

        return torch.Tensor(item[:self.label_col] + item[self.label_col+1:]), torch.Tensor([item[self.label_col]])
