import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader with inverse-frequency sampling and replacement.
    """
    # count each label's occurrency
    class_p = 1/torch.bincount(labels)
    weights = class_p[labels]
    

    sampler = WeightedRandomSampler(weights, weights.numel(), replacement=True)

    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)