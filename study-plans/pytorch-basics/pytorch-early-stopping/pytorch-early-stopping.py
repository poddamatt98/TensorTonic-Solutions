import torch
import torch.nn as nn
import copy

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    train_loss_hist = []
    val_loss_hist = []

    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    best_weights = None

    def build_result(epoch, train_losses, val_losses):
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'stopped_epoch': epoch
        }

    

    for epoch in range(max_epochs):
        if epoch > patience:
            if best_val_loss_epoch < (epoch - patience):
                return build_result(epoch, train_loss_hist, val_loss_hist)
            
        # train
        model.train()

        epoch_train_cum_loss = 0
        epoch_train_batches = 0
        for x, y in train_loader:
            out = model(x)
            loss = criterion(y, out).mean()
            epoch_train_batches +=1
            epoch_train_cum_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_hist.append(epoch_train_cum_loss / epoch_train_batches)

        # validation
        model.eval()
        with torch.no_grad():
            epoch_val_batches = 0
            epoch_val_cum_loss = 0
            for x, y in val_loader:
                out = model(x)
                loss = criterion(y, out).mean()
                epoch_val_batches += 1
                epoch_val_cum_loss += loss.item()
            val_loss_hist.append(epoch_val_cum_loss)
            if epoch_val_cum_loss < best_val_loss:
                best_val_loss = epoch_val_cum_loss
                best_val_loss_epoch = epoch

    return build_result(epoch+1, train_loss_hist, val_loss_hist)
            
            
