from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
OUTS_BASE_DIR = REPO_ROOT / "outs"
RANDOM_SEED = 0


class Training:
    batch_size: int = 128
    n_data_loader_workers: int = 4
    learning_rate: float = 1e-3
    n_epochs: int = 5