import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from unet import DeclippingUNet, center_crop_1d, prepare_unet_input
from pathlib import Path


# https://arxiv.org/pdf/2602.22279
def loss_mc(y, model, mu):
    x_hat = model(y)
    y = center_crop_1d(y, x_hat.size(-1))
    zeros = torch.zeros_like(x_hat)
    diff = y - x_hat
    rho = torch.where(y.abs() < mu, torch.abs(diff), zeros)
    rho = rho + torch.where(y == mu, torch.relu(diff), zeros)
    rho = rho + torch.where(y == - mu, torch.relu(-diff), zeros)
    return rho.square().sum()

# https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000852.pdf
def measurement_consistency_loss(x_hat, y, threshold, eps=1e-6):
    y = center_crop_1d(y, x_hat.size(-1))
    cond_mask = (y.abs() < threshold) | (torch.sign(y) * x_hat < threshold)
    # cond_mask = (y.abs() < threshold - eps) | (torch.sign(y) * x_hat < threshold)
    h = torch.where(cond_mask, y - x_hat, torch.zeros_like(y))
    return torch.sum(h ** 2)


def equivariance_loss(x_hat, model, threshold, g_min = 0.1, g_max = 2.0):
    g_shape = [x_hat.shape[0]] + [1] * (x_hat.ndim - 1)
    # create empty tensor shaped like x_hat, fill it with random numbers between g_min and g_max
    g = torch.empty(g_shape, device=x_hat.device, dtype=x_hat.dtype).uniform_(g_min, g_max)

    gx_hat = g * x_hat
    gx_hat = prepare_unet_input(gx_hat)

    y_g = torch.clamp(gx_hat, -threshold, threshold)
    gx_hat_reconstructed = model(y_g)

    gx_hat = center_crop_1d(
        gx_hat,
        gx_hat_reconstructed.size(-1),
    )
    return torch.mean((gx_hat - gx_hat_reconstructed) ** 2)

def total_loss(x_hat, y, model, threshold, eps=1e-6, g_min=0.1, g_max=2.0, lambda_w=1.0):
    L_mc = measurement_consistency_loss(x_hat, y, threshold, eps)
    L_ei = equivariance_loss(x_hat, model, threshold, g_min, g_max)
    return L_mc + lambda_w * L_ei, L_mc, L_ei



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_dir = Path(__file__).resolve().parent.parent
    path = (project_dir / "dataset")
    dataset = AudioDataset(path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    x = next(iter(dataloader)).to(device)
    threshold = 0.1
    y = torch.clamp(x, -threshold, threshold)
    model = DeclippingUNet().to(device)
    x_hat = model(y)

    # g range used for GTZAN samples
    gmin = 0.1
    gmax = 2.0
    eps = 1e-6
    lambda_w = 0.1

    loss, loss_mc, loss_ei = total_loss(x_hat, y, model, threshold, eps, gmin, gmax, lambda_w)
    print(f"loss_mc = {loss_mc}")
    print(f"loss_ei = {loss_ei}")
    print(f"loss_total = {loss}")