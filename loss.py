import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

path = "dataset"
dataset = AudioDataset(path)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

n_sigs = 5
thresholds = np.arange(0.0001, 0.001, 0.01)
threshold = 0.0001

g = 2

def f_0(x):
    xscale = 1.5 * x
    return xscale

for i in range(n_sigs):
    print(f"Input: {i+1}, Threshold: {threshold}")
    input_sig = next(iter(dataloader))
    # y = torch.clip(input_sig, -threshold, threshold).to(device).requires_grad_(True)
    y = torch.clip(input_sig, -threshold, threshold)

    L_ei = g * f_0(y) - f_0(torch.clip(g*f_0(y), -threshold, threshold))






    L_mc = torch.mean(torch.abs(y))
    print(f"L_mc: {L_mc}")

    L = L_mc + L_ei
    print(f"L: {L}")

# def clip_signal(x, threshold) -> torch.Tensor:
#     return torch.clamp(x, min=-threshold, max=threshold)

def equivariance_loss(
    model: nn.Module,
    y: torch.Tensor,
    threshold: = 0.1,
    g_min: = 0.1,
    g_max: = 2.0,
) -> torch.Tensor:

    x_hat = model(y)  # f_theta(y)

    # Shape is [B, 1, 1] for multichannel audio or [B, 1] for mono batches.
    g_shape = [y.shape[0]] + [1] * (y.ndim - 1)
    g = torch.empty(g_shape, device=y.device, dtype=y.dtype).uniform_(g_min, g_max)

    scaled_x_hat = g * x_hat                        # g * f_theta(y)
    # remeasured = clip_signal(scaled_x_hat, threshold)  # eta(g * f_theta(y))
    remeasured = torch.clamp(scaled_x_hat, -threshold, threshold)  # eta(g * f_theta(y))
    reconstructed_remeasured = model(remeasured)    # f_theta(eta(g * f_theta(y)))

    return torch.mean((scaled_x_hat - reconstructed_remeasured) ** 2)


def measurement_consistency_loss(
    x_hat: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.1,
    atol: float = 1e-6,
) -> torch.Tensor:
    """
    Measurement consistency loss L_MC from Eq. (13).

    For unsaturated samples, it enforces x_hat = y.
    For positive saturated samples (+mu), it penalizes only x_hat < +mu.
    For negative saturated samples (-mu), it penalizes only x_hat > -mu.

    Args:
        x_hat: Reconstructed audio f_theta(y).
        y: Observed clipped audio.
        threshold: Clipping threshold mu.
        atol: Tolerance used to identify saturated samples.

    Returns:
        Scalar mean squared consistency loss.
    """
    is_positive_clipped = y >= (threshold - atol)
    is_negative_clipped = y <= (-threshold + atol)
    is_unsaturated = ~(is_positive_clipped | is_negative_clipped)

    h = torch.zeros_like(y)

    # If |y| < mu, require reconstruction exactly equal to observation.
    h[is_unsaturated] = y[is_unsaturated] - x_hat[is_unsaturated]

    # For y = +mu, reconstruction is valid if x_hat >= +mu.
    h[is_positive_clipped] = torch.minimum(
        torch.zeros_like(x_hat[is_positive_clipped]),
        x_hat[is_positive_clipped] - threshold,
    )

    # For y = -mu, reconstruction is valid if x_hat <= -mu.
    h[is_negative_clipped] = torch.maximum(
        torch.zeros_like(x_hat[is_negative_clipped]),
        x_hat[is_negative_clipped] + threshold,
    )

    return torch.mean(h ** 2)


def self_supervised_declipping_loss(
    model: nn.Module,
    y: torch.Tensor,
    threshold: float = 0.1,
    g_min: float = 0.1,
    g_max: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Complete loss L = L_MC + L_EI.

    Returns:
        total_loss, measurement_consistency_loss, equivariance_loss
    """
    x_hat = model(y)

    loss_mc = measurement_consistency_loss(
        x_hat=x_hat,
        y=y,
        threshold=threshold,
    )

    loss_ei = equivariance_loss(
        model=model,
        y=y,
        threshold=threshold,
        g_min=g_min,
        g_max=g_max,
    )

    total_loss = loss_mc + loss_ei
    return total_loss, loss_mc, loss_ei