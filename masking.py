import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import DeclippingModel

def apply_mask(x_hat, y, tau = 0.95, mu = 0.1):
    zero = torch.zeros((), dtype=y.dtype, device=y.device)
    b = torch.maximum(zero, torch.abs(y) - tau * mu) / (1.0 - tau * mu)
    b = torch.clamp(b, 0.0, 1.0)  # b musi zustat v [0,1], jinak extrapolace mimo interval
    x_masked = (1.0 - b) * y + b * x_hat
    return x_masked

if __name__ == "__main__":
    path = "dataset"
    dataset = AudioDataset(path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    x = next(iter(dataloader))
    threshold = 0.1
    y = torch.clamp(x, -threshold, threshold)
    model = DeclippingModel()
    x_hat = model(y)

    # y = torch.tensor([0.2, 0.5, 0.94, 1.0, -1.0, -0.5, 0.99])
    # x_hat = torch.tensor([0.25, 0.55, 1.10, 1.30, -1.25, -0.45, 1.05])
    mu = 1.0

    result = apply_mask(x_hat, y, mu=mu, tau=0.95)

    for yi, xi, ri in zip(y.tolist(), x_hat.tolist(), result.tolist()):
        print(f"y={yi}  x_hat={xi}  ->  x_masked={ri}")
