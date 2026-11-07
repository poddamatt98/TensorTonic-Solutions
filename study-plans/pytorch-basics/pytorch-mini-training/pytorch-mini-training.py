import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    epoch_loss = 0
    num_batches = 0
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)

        loss = criterion(y, out)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    return epoch_loss/num_batches
