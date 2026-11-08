import torch


def get_checkpoints(run_dir, training_run, wandb_run_id):
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")
    checkpoints = {
        'best': checkpoints_dir / f"{training_run}_{wandb_run_id}_best.pt",
        'last': checkpoints_dir / f"{training_run}_{wandb_run_id}_last.pt",
        'final': checkpoints_dir / f"declipping_{training_run}_{wandb_run_id}.pt",
    }
    for name, checkpoint_path in checkpoints.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint '{name}' not found: {checkpoint_path}")
    return checkpoints

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        return (checkpoint["model_state_dict"], checkpoint.get("config", {}), checkpoint.get("epoch"))
    return checkpoint, {}, None


if __name__ == "__main__":
    checkpoints_dir = "checkpoints"
    # checkpoints = get_ckpt_groups(checkpoints_dir)