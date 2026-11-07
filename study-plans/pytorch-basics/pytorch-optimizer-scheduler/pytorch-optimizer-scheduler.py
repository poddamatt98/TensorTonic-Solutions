import torch
import torch.nn as nn

def train_with_scheduler(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR, num_epochs: int) -> dict:
    """
    Returns losses and lrs as lists of Python floats in a dictionary.
    """

    lr_history = []
    loss_history = []
    
    model.train()

    for epoch in range(num_epochs):
        
        lr_history.append(scheduler.get_last_lr()[0])
        epoch_cumulative_loss = 0
        epoch_steps = 0
        for x, y in dataloader:
            out = model(x)
            loss = criterion(y, out)
            epoch_cumulative_loss += loss.item()
            epoch_steps += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss_history.append(epoch_cumulative_loss / epoch_steps)

    return {
        'losses': loss_history,
        'lrs': lr_history
    }
        
        
