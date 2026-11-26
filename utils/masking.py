import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from unet import DeclippingUNet, center_crop_1d, prepare_unet_input

def apply_mask(x_hat, y, tau = 0.95, threshold = 0.1, blending=False):
    if x_hat.shape != y.shape:
        y = center_crop_1d(y, x_hat.size(-1))
        # raise ValueError(f"x_hat and y must have identical shapes, got {tuple(x_hat.shape)} and {tuple(y.shape)}.")

    # https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000852.pdf
    if blending:
        zero = torch.zeros((), dtype=y.dtype, device=y.device)
        b = torch.maximum(zero, torch.abs(y) - tau * threshold) / (1.0 - tau * threshold)
        # b = torch.clamp(b, 0.0, 1.0)  # b musi zustat v [0,1], jinak extrapolace mimo interval
        print(b)
        x_masked = (1.0 - b) * y + b * x_hat
        return x_masked

    # https://arxiv.org/pdf/2602.22279
    else:
        mask = y.abs() >= threshold - 1e-4
        return torch.where(mask, x_hat, y)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "../dataset"
    dataset = AudioDataset(path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = DeclippingUNet()
    threshold = 0.1

    # x = next(iter(dataloader)).to(device)
    # x = prepare_unet_input(x)
    # y = torch.clamp(x, -threshold, threshold).to(device)
    # x_hat = model(y)

    y = torch.tensor([0.2, 0.5, 0.94, 1.0, -1.0, -0.5, 0.99])
    x_hat = torch.tensor([0.25, 0.55, 1.10, 1.30, -1.25, -0.45, 1.05])
    result = apply_mask(x_hat=x_hat, y=y, threshold=threshold, blending=False)

    for yi, xi, ri in zip(y.tolist(), x_hat.tolist(), result.tolist()):
        print(f"y={yi}  x_hat={xi}  ->  x_masked={ri}")
