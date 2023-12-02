from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
OUTS_BASE_DIR = REPO_ROOT / "outs"
RANDOM_SEED = 0


class Training:
    batch_size: int = 128
    cluster_loss_weight: float = 0.8
    learning_rate: float = 1e-3
    n_classes: int = 10
    n_data_loader_workers: int = 4
    n_epochs: int = 100
    n_prototypes: int = 30
    n_slots_per_class: int = 2
    separation_loss_weight: float = -0.08
