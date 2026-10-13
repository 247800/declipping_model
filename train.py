import os
import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from unet import DeclippingUNet
from loss import total_loss
import wandb

def main():
    wandb.login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using", device)
    torch.manual_seed(0)

    # hyperparameters following section 5.3 of https://arxiv.org/pdf/2602.22279
    lambda_weight = 0.1   # weight of equivariance loss
    gmin = 0.1
    gmax = 2.0
    learning_rate = 5e-4
    epochs = 100
    batch_size = 35
    threshold = 0.1

    dataset_dir = "dataset"
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    dataset = AudioDataset(directory=dataset_dir, clipping_threshold=threshold)
    if len(dataset) == 0:
        raise RuntimeError("Training dataset contains no segments.")
    dataset.print_params()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = DeclippingUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run = wandb.init(
        project="declipping-unet-train",
        name="first-training",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "threshold": threshold,
            "lambda_weight": lambda_weight,
            "g_min": gmin,
            "g_max": gmax,
            "num_segments": len(dataset),
        },
    )

    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        sum_loss = sum_mc = sum_ei = 0.0
        n_batches = 0

        for x in dataloader:
            x = x.to(device, non_blocking=True)    # [B, 1, 22050]
            y = torch.clamp(x, -threshold, threshold)

            optimizer.zero_grad()
            x_hat = model(y)

            loss, loss_mc, loss_ei = total_loss(
                x_hat=x_hat,
                y=y,
                model=model,
                threshold=threshold,
                g_min=gmin,
                g_max=gmax,
                lambda_w=lambda_weight,
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            sum_loss += loss.item()
            sum_mc += loss_mc.item()
            sum_ei += loss_ei.item()
            n_batches += 1

            wandb.log({
                "train/step_loss": loss.item(),
                "train/step_loss_mc": loss_mc.item(),
                "train/step_loss_ei": loss_ei.item(),
            }, step=global_step)

        avg_loss = sum_loss / n_batches
        avg_mc = sum_mc / n_batches
        avg_ei = sum_ei / n_batches

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "train/loss_mc": avg_mc,
            "train/loss_ei": avg_ei,
        }, step=global_step)

        print(f"Epoch [{epoch + 1}/{epochs}]  "
              f"loss: {avg_loss:.6f}  loss_mc: {avg_mc:.6f}  loss_ei: {avg_ei:.6f}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": dict(run.config),
        }

        # save last checkpoint
        torch.save(checkpoint, os.path.join(weights_dir, f"{run.name}_{run.id}_last.pt"))

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(weights_dir, f"{run.name}_{run.id}_best.pt"))

    # save final weights
    torch.save(model.state_dict(), os.path.join(weights_dir, f"declipping_{run.name}_{run.id}.pt"))

    wandb.finish()


if __name__ == "__main__":
    main()