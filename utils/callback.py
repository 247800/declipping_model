import wandb
import torchaudio
import soundfile as sf
from pathlib import Path
import torch
import re
from datetime import datetime
from uuid import uuid4
import json


def create_run_dirs(root, label):
    """Create exclusive directory for one training run"""
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        uuid_code = uuid4().hex[:4]
        run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{safe_label}_{uuid_code}"
        run_dir = root / run_name
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    paths = {"run": run_dir}
    for name in ("checkpoints", "audio", "logs", "metadata"):
        paths[name] = run_dir / name
        paths[name].mkdir()

    return run_name, uuid_code, paths

def get_sr(directory, expected_sr=None):
    files = sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        raise FileNotFoundError(f"No WAV files found in {directory!r}")
    _, loaded_sr = torchaudio.load(str(files[0]))
    if expected_sr is not None and loaded_sr != expected_sr:
        raise ValueError(f"{files[0]}: expected sample rate of {expected_sr} but got {loaded_sr} instead")
    return loaded_sr

def wb_run(run_name, learning_rate, epochs, batch_size, threshold, lambda_weight, gmin, gmax, dataset, logs_dir):
    wandb.login()
    run = wandb.init(
        project="declipping-unet-training",
        name=run_name,
        job_type="training",
        dir=str(logs_dir),
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "threshold": threshold,
            "lambda_weight": lambda_weight,
            "g_min": gmin,
            "g_max": gmax,
            "num_segments": len(dataset),
            "split_seed": dataset.seed,
            "torch_seed": 0,
        },
    )
    # Use custom axes
    run.define_metric("global_step")
    run.define_metric("epoch")
    run.define_metric("train/step_*", step_metric="global_step")
    run.define_metric("train/epoch_*", step_metric="epoch")
    return run

def wb_log_step(run, loss, loss_mc, loss_ei, global_step):
    run.log({
        "global_step": global_step,
        "train/step_loss": loss,
        "train/step_loss_mc": loss_mc,
        "train/step_loss_ei": loss_ei,
    })

def wb_log_epoch(run, epoch, avg_loss, avg_mc, avg_ei, global_step):
    run.log({
        "epoch": epoch,
        "global_step": global_step,
        "train/loss": avg_loss,
        "train/loss_mc": avg_mc,
        "train/loss_ei": avg_ei,
    })

def export_audio(output_dir, wf, sr, epoch, global_step, uuid_code):
    if wf.ndim != 2:
        raise ValueError("Expected waveform with shape [channels, samples].")

    audio = wf.detach().to(device="cpu", dtype=torch.float32)

    if not torch.isfinite(audio).all().item():
        raise ValueError("Cannot export non-finite audio samples.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (f"{uuid_code}_epoch{epoch:03d}_step{global_step:07d}_reconstructed.wav")
    path = output_dir / filename

    # Exclusive mode protects existing files even inside the same run
    with path.open("xb") as stream:
        sf.write(stream, audio.transpose(0, 1).numpy(), sr, format="WAV", subtype="FLOAT")

    print(f"Audio exported: {path}")
    return path

def save_json(path, data):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)

