import torch

class TransformPipeline:
    def __init__(self, mean: list, std: list):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns a normalized float32 tensor with shape (C, H, W).
        """
        normalized = (image/255 - self.mean) / self.std
        return normalized.permute((2, 0, 1))
        
