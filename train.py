import os
import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from unet import DeclippingUNet
from utils.loss import total_loss
from utils import callback as cb
from pathlib import Path

def main():
    project_dir = Path(__file__).resolve().parent
    run_name, uuid_code, paths = cb.create_run_dirs(
        root=project_dir / "exports_train",
        label="gtzan100", # adjust the label to the specific run; gtzan100 refers to 100 files from gtzan dataset
    )
    audio_dir = paths["audio"]
    json_dir = paths["metadata"]
    checkpoint_dir = paths["checkpoints"]
    logs_dir = paths["logs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # "cuda:0"
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
    sr_expected = 22050   # sample rate of GTZAN files
    sample_rate = cb.get_sr(dataset_dir, sr_expected)
    dataset = AudioDataset(directory=dataset_dir, sr=sample_rate, clipping_threshold=threshold)
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

    run = cb.wb_run(run_name, learning_rate, epochs, batch_size, threshold, lambda_weight, gmin, gmax, dataset, logs_dir)  # project="declipping-unet-train", job_type="training"
    cb.save_json(json_dir / "train_files.json", dataset.files)

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
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

            if global_step % 50 == 0:
                cb.wb_log_step(run, loss.item(),loss_mc.item(), loss_ei.item(), global_step)
                cb.export_audio(audio_dir, x_hat[0], sample_rate, epoch, global_step, uuid_code)

        avg_loss = sum_loss / n_batches
        avg_mc = sum_mc / n_batches
        avg_ei = sum_ei / n_batches

        if epoch % 10 == 0:
            cb.wb_log_epoch(run, epoch, avg_loss, avg_mc, avg_ei, global_step)

        print(f"Epoch [{epoch:3d}/{epochs}] loss: {avg_loss:.6f}  loss_mc: {avg_mc:.6f}  loss_ei: {avg_ei:.6f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": dict(run.config),
        }

        # save last checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"{run.name}_{run.id}_last.pt"))

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"{run.name}_{run.id}_best.pt"))

    # save final checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"declipping_{run.name}_{run.id}.pt"))

    run.finish()


if __name__ == "__main__":
    main()