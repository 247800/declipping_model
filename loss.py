import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from unet import DeclippingUNet as DeclippingModel

def center_crop_1d(x, target_length) -> torch.Tensor:
    """
    Due to no padding in Unet's convolutions the model output size is smaller than its input.
    To compute loss, both signals compared must be of the same length while preserving
    the valid part of the cropped signal.
    """
    current_length = x.size(-1)
    if current_length < target_length:
        raise ValueError(
            f"Cannot crop length {current_length} "
            f"to larger target length {target_length}."
        )
    difference = current_length - target_length
    left = difference // 2
    right = left + target_length
    return x[:, :, left:right]

def measurement_consistency_loss(x_hat, y, threshold, eps=1e-6):
    y = center_crop_1d(y, x_hat.size(-1))
    cond_mask = (y.abs() < threshold - eps) | (torch.sign(y) * x_hat < threshold)
    h = torch.where(cond_mask, y - x_hat, torch.zeros_like(y))
    return torch.mean(h ** 2)

def equivariance_loss(x_hat, model, threshold, g_min = 0.1, g_max = 2.0):
    g_shape = [x_hat.shape[0]] + [1] * (x_hat.ndim - 1)
    g = torch.empty(g_shape, device=x_hat.device, dtype=x_hat.dtype).uniform_(g_min, g_max)
    gx_hat = g * x_hat
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
    path = "dataset"
    dataset = AudioDataset(path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    x = next(iter(dataloader)).to(device)
    threshold = 0.1
    y = torch.clamp(x, -threshold, threshold)
    model = DeclippingModel().to(device)
    x_hat = model(y)

    # g range used for GTZAN samples
    gmin = 0.1
    gmax = 2.0

    # g range used for synthetic dataset samples
    # gmin = 0.5
    # gmax = 1.5

    eps = 1e-6
    lambda_w = 0.1

    loss, loss_mc, loss_ei = total_loss(x_hat, y, model, threshold, eps, gmin, gmax, lambda_w)
    print(f"loss_mc = {loss_mc}")
    print(f"loss_ei = {loss_ei}")
    print(f"loss_total = {loss}")