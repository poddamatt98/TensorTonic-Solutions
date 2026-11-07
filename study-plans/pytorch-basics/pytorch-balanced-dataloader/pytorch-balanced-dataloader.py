import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    """
    Returns a DataLoader with inverse-frequency sampling and replacement.
    """
    # count each label's occurrency
    weights = torch.zeros(labels.shape)
    for label in labels.unique():
        label_count = torch.sum(labels == label)
        label_weight = 1/label_count
        weights += label_weight * (labels == label)

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)