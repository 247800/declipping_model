import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import DeclippingModel

def measurement_consistency_loss(x_hat, y, threshold, eps=1e-6):
    # eps = tolerance při zaokrouhlování floatu
    cond_mask = (y.abs() < threshold - eps) | (torch.sign(y) * x_hat < threshold)
    h = torch.where(cond_mask, y - x_hat, torch.zeros_like(y))
    return torch.mean(h ** 2)

def equivariance_loss(x_hat, model, threshold, g_min = 0.1, g_max = 2.0):
    g_shape = [x_hat.shape[0]] + [1] * (x_hat.ndim - 1)
    g = torch.empty(g_shape, device=x_hat.device, dtype=x_hat.dtype).uniform_(g_min, g_max)
    x_g = g * x_hat
    y_g = torch.clamp(x_g, -threshold, threshold)
    x_g_hat = model(y_g)
    return torch.mean((x_g - x_g_hat) ** 2)

def total_loss(x_hat, y, model, threshold, eps=1e-6, g_min=0.1, g_max=2.0):
    L_mc = measurement_consistency_loss(x_hat, y, threshold, eps)
    L_ei = equivariance_loss(x_hat, model, threshold, g_min, g_max)
    return L_mc + L_ei, L_mc, L_ei


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

path = "dataset"
dataset = AudioDataset(path)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

x = next(iter(dataloader))
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

loss, loss_mc, loss_ei = total_loss(x_hat, y, model, threshold, eps, gmin, gmax)
print(f"loss_mc = {loss_mc}")
print(f"loss_ei = {loss_ei}")
print(f"loss_total = {loss}")