import torch

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        defaults= {'lr': lr, 'momentum': momentum}
        super().__init__(params, defaults)
        print(self.param_groups)

        self.v_last = [[torch.zeros_like(p) for p in group['params']] for group in self.param_groups]

    def step(self, closure=None):
        """
        Returns the closure loss when provided, otherwise None.
        """
        closure_val = None
        if closure:
            closure_val = closure()

        for group, v_lasts in zip(self.param_groups, self.v_last):
            with torch.no_grad():
                lr = group['lr']
                momentum = group['momentum']
                for param, v_last in zip(group['params'], v_lasts):
                    if param.grad is not None:
                        v = momentum * v_last + param.grad
                        param -= lr * v
                        v_last.copy_(v)

        return closure_val