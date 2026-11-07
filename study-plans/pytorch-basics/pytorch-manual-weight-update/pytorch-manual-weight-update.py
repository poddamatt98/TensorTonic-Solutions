import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """
    model.train()

    
    for param in model.parameters():
        param.requires_grad_(True)

    out = model(X)
    loss = criterion(y, out)
    loss.backward()
    for param in model.parameters():
        with torch.no_grad():
            param -= lr * param.grad
        param.grad.zero_()
    
    return loss.item()