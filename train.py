import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import AudioDataset
from unet import DeclippingUNet, prepare_unet_input, center_crop_1d
from utils.loss import total_loss
from utils import callback as cb
from utils.masking import apply_mask
from torchmetrics.functional.audio import signal_noise_ratio
import torch.nn as nn


@torch.no_grad()
def evaluate(model, test_dataloader, threshold, device):
    was_training = model.training
    model.eval()
    out_snrs, in_snrs = [], []
    try:
        for x in test_dataloader:
            x = prepare_unet_input(x.to(device, non_blocking=True))
            y = torch.clamp(x, -threshold, threshold)
            x_hat = model(y)
            y_cropped = center_crop_1d(y, x_hat.size(-1))
            x_cropped = center_crop_1d(x, x_hat.size(-1))
            x_hat_masked = apply_mask(x_hat=x_hat, y=y_cropped, threshold=threshold)
            out_snrs.append(signal_noise_ratio(preds=x_hat_masked, target=x_cropped, zero_mean=False))
            in_snrs.append(signal_noise_ratio(preds=y_cropped, target=x_cropped, zero_mean=False))
        if not out_snrs:
            raise RuntimeError("Evaluation loader contains no batches.")
        out_snr = torch.cat(out_snrs).mean().item()
        in_snr = torch.cat(in_snrs).mean().item()
        return out_snr, in_snr, out_snr - in_snr
    finally:
        model.train(was_training)


def homogeneity_error(model: nn.Module, y: torch.Tensor, gains=(0.1, 0.5, 2.0, 10.0)) -> dict:
    errs = {}
    with torch.no_grad():
        fy = model(y)
        for g in gains:
            ref = g * fy
            errs[g] = ((model(g * y) - ref).norm() / ref.norm()).item()
    return errs


def main():
    project_dir = Path(__file__).resolve().parent
    run_name, uuid_code, paths = cb.create_run_dirs(
        root=project_dir / "exports_train",
        label="unet2-lrl_lw01_mc-sum_ei-mean", # adjust the label to the specific run
    )
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

    dataset_dir = project_dir / "genres_original"
    sr_expected = 22050   # sample rate of GTZAN files
    sample_rate = cb.get_sr(dataset_dir, sr_expected)

    # training subset
    dataset = AudioDataset(
        directory=dataset_dir,
        sr=sample_rate,
        clipping_threshold=threshold,
        filter_unsaturated=True,
        train=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("Training dataset contains no segments.")
    dataset.print_params()

    # evaluating subset
    test_dataset = AudioDataset(
        directory=dataset_dir,
        sr=sample_rate,
        clipping_threshold=threshold,
        filter_unsaturated=False,
        train=False
    )
    if len(test_dataset) == 0:
        raise RuntimeError("Testing dataset contains no segments.")
    test_dataset.print_params()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = DeclippingUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run = cb.wb_run(run_name, learning_rate, epochs, batch_size, threshold, lambda_weight, gmin, gmax, dataset, logs_dir)  # project="declipping-unet-train", job_type="training"
    cb.save_json(json_dir / "train_files.json", dataset.files)
    cb.save_json(json_dir / "test_files.json", test_dataset.files)

    model.eval()
    test_input = prepare_unet_input(0.1 * torch.randn(2, 22050, device=device))
    errs = homogeneity_error(model, test_input)
    print("[model] homogeneity rel. error: " + "  ".join(f"g={g}: {e:.1e}" for g, e in errs.items()))

    global_step = 0
    detach_ei = True
    best_loss = float("inf")
    best_delta = float("-inf")

    example_x = next(iter(test_dataloader))
    example_x = prepare_unet_input(example_x)
    example_x = example_x[:8].to(device)
    example_y = torch.clamp(example_x, -threshold, threshold)

    _, in_snr, d0 = evaluate(model, test_dataloader, threshold, device)
    print(f"[eval] untrained model: input SNR {in_snr:.2f} dB, identity delta SNR {d0:.3f} dB")

    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = sum_mc = sum_ei = 0.0
        n_batches = 0

        for x in dataloader:
            x = x.to(device, non_blocking=True)
            x = prepare_unet_input(x)
            y = torch.clamp(x, -threshold, threshold).to(device)
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
                detach_ei=detach_ei,
            )

            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_mc += loss_mc.item()
            sum_ei += loss_ei.item()
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                cb.wb_log_step(run, loss.item(),loss_mc.item(), loss_ei.item(), global_step)

            if global_step == 1:
                cb.export_audio(paths["original"], x[0],sample_rate, epoch, global_step, uuid_code,"original")
                # cb.export_audio(paths["clipped"], y[0], sample_rate, epoch, global_step, uuid_code, "clipped")
                cb.export_audio(paths["reconstructed"], x_hat[0], sample_rate, epoch, global_step, uuid_code, "reconstructed")

        avg_loss = sum_loss / n_batches
        avg_mc = sum_mc / n_batches
        avg_ei = sum_ei / n_batches
        cb.wb_log_epoch(run, epoch, avg_loss, avg_mc, avg_ei, global_step)

        out_snr, in_snr, delta_snr = evaluate(model, test_dataloader, threshold, device)
        print(f"epoch {epoch}: mc {sum_mc / n_batches:.4g}  ei {sum_ei / n_batches:.4g} | "
              f"SNR {out_snr:.2f} dB (input {in_snr:.2f} dB, dSNR {delta_snr:+.2f} dB)")
        cb.wb_log_eval(run=run, epoch=epoch, out_snr=out_snr, in_snr=in_snr, delta_snr=delta_snr, global_step=global_step)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                example_x_hat = model(example_y)
                output_length = example_x_hat.size(-1)

                example_y_cropped = center_crop_1d(example_y, output_length)
                example_x_cropped = center_crop_1d(example_x, output_length)
                example_masked = apply_mask(x_hat=example_x_hat, y=example_y_cropped, threshold=threshold)

                cb.export_audio(paths["example"], example_masked[0], sample_rate, epoch, global_step, uuid_code,"example_masked")
                cb.export_audio(paths["example"], example_x_cropped[0], sample_rate, epoch, global_step, uuid_code, "example_clean")
        finally:
            model.train(was_training)

        is_best_loss = avg_loss < best_loss
        is_best_delta = delta_snr > best_delta

        if is_best_loss:
            best_loss = avg_loss

        if is_best_delta:
            best_delta = delta_snr

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "loss_mc": avg_mc,
            "loss_ei": avg_ei,
            "detach_ei": detach_ei,
            "val_snr_out_db": out_snr,
            "val_snr_in_db": in_snr,
            "val_delta_snr_db": delta_snr,
            "best_loss": best_loss,
            "best_delta": best_delta,
            "config": dict(run.config),
        }

        if epoch % 10 == 0:
            cb.export_audio(paths["original"], x[0], sample_rate, epoch, global_step, uuid_code, "original")
            # cb.export_audio(paths["clipped"], y[0], sample_rate, epoch, global_step, uuid_code, "clipped")
            cb.export_audio(paths["reconstructed"], x_hat[0], sample_rate, epoch, global_step, uuid_code,"reconstructed")
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"{run.name}_{run.id}_epoch{epoch:03d}.pt"))

        # save best by avg_loss
        if is_best_loss:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"{run.name}_{run.id}_best-avg.pt"))

        # save best by snr improvement
        if is_best_delta:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"{run.name}_{run.id}_best-delta.pt"))

        # save last checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"{run.name}_{run.id}_last.pt"))

        print(f"Epoch [{epoch:03d}/{epochs}] loss: {avg_loss:.6f}  loss_mc: {avg_mc:.6f}  loss_ei: {avg_ei:.6f}")

    # save final checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{run.name}_{run.id}_final.pt"))

    run.finish()


if __name__ == "__main__":
    main()