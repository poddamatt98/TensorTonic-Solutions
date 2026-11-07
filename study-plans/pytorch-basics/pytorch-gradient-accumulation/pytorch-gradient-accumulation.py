import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """

    weights = w_init
    weights.requires_grad_(True)


    step_count = 0
    gradients = torch.zeros_like(w_init)
    mean_grad = 0
    for x, y in micro_batches:
        step_count += 1
        out = weights.T@torch.Tensor(x)
        loss = torch.square(out - torch.Tensor([y]))

        loss.backward()
        with torch.no_grad():
            if step_count % accum_steps == 0:

                # store the mean grad
                mean_grad = weights.grad / accum_steps
                # store the mean grad
                weights -= lr* mean_grad       
                # zero grad
                weights.grad.zero_()
                


    return weights, mean_grad
    
            
        
    
