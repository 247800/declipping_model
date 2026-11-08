import wandb
import soundfile as sf
from pathlib import Path
import torch
import re
from datetime import datetime
import json
import math


def create_run_dirs(root, label, uuid_code, thresholds):
    """
    Create exclusive directory for one testing run.
    Label refers to dataset used for testing.
    Uuid code is passed from corresponding training run to identify checkpoint version.
    """
    thresholds = list(thresholds)
    if not thresholds:
        raise ValueError("At least one clipping threshold is required.")
    if any(not math.isfinite(threshold) or threshold <= 0 for threshold in thresholds):
        raise ValueError("Clipping thresholds must be finite positive numbers.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Clipping thresholds must be unique.")

    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{safe_label}_{uuid_code}"
        run_dir = root / run_name
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    paths = {"run": run_dir}
    for name in ("audio", "logs", "metadata"):
        paths[name] = run_dir / name
        paths[name].mkdir()

    paths["checkpoints"] = {}
    for checkpoint_name in ("best", "last", "final"):
        checkpoint_paths = {}
        paths["checkpoints"][checkpoint_name] = checkpoint_paths

        for threshold in thresholds:
            threshold_name = (f"thr_{str(threshold).replace('.', 'p')}")
            audio_dir = (paths["audio"] / f"ckpt_{checkpoint_name}" / threshold_name)
            metadata_dir = (paths["metadata"] / f"ckpt_{checkpoint_name}" / threshold_name)
            metadata_dir.mkdir(parents=True, exist_ok=False)
            threshold_paths = {"metadata": metadata_dir}

            for audio_type in ("original", "clipped", "reconstructed","masked"):
                audio_path = audio_dir / audio_type
                audio_path.mkdir(parents=True, exist_ok=False)
                threshold_paths[audio_type] = audio_path

            checkpoint_paths[threshold] = threshold_paths
    return run_name, uuid_code, paths

def get_sr(directory, expected_sr=None):
    files = sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        raise FileNotFoundError(f"No WAV files found in {directory!r}")
    loaded_sr = sf.info(str(files[0])).samplerate
    if expected_sr is not None and loaded_sr != expected_sr:
        raise ValueError(f"{files[0]}: expected sample rate of {expected_sr} but got {loaded_sr} instead")
    return loaded_sr

def wb_run_test(run_name, batch_size, thresholds, dataset, logs_dir):
    wandb.login()
    run = wandb.init(
        project="declipping-unet-test",
        name=run_name,
        job_type="test",
        dir=str(logs_dir),
        config={
            "batch_size": batch_size,
            "thresholds": thresholds,
            # "lambda_weight": lambda_weight,
            # "g_min": gmin,
            # "g_max": gmax,
            "num_segments": len(dataset),
            "split_seed": dataset.seed,
            "torch_seed": 0,
        },
    )
    run.define_metric("step")
    run.define_metric("threshold")
    run.define_metric("test/step_*", step_metric="step")
    run.define_metric("test/threshold_*", step_metric="threshold")
    return run

def wb_log_step(run, loss, loss_mc, loss_ei, step):
    """
    Log loss for one step within one threshold.
    """
    run.log({
        "step": step,
        "test/step_loss": loss,
        "test/step_loss_mc": loss_mc,
        "test/step_loss_ei": loss_ei,
    })

def wb_log_avg(run, threshold, avg_loss, avg_mc, avg_ei, step):
    """
    Log average loss for one threshold within one checkpoint.
    """
    run.log({
        "threshold": threshold,
        "step": step,
        "test/loss": avg_loss,
        "test/loss_mc": avg_mc,
        "test/loss_ei": avg_ei,
    })

def export_audio(output_dir, wf, sr, threshold, step, uuid_code, name="reconstructed"):
    if wf.ndim != 2:
        raise ValueError("Expected waveform with shape [channels, samples].")

    audio = wf.detach().to(device="cpu", dtype=torch.float32)

    if not torch.isfinite(audio).all().item():
        raise ValueError("Cannot export non-finite audio samples.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (f"{uuid_code}_threshold{threshold:03d}_step{step:03d}_{name}.wav")
    path = output_dir / filename

    # Exclusive mode protects existing files even inside the same run
    with path.open("xb") as stream:
        sf.write(stream, audio.transpose(0, 1).numpy(), sr, format="WAV", subtype="FLOAT")

    print(f"Audio exported: {path}")
    return path

def save_json(path, data):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)

