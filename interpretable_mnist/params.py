from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
OUTS_BASE_DIR = REPO_ROOT / "outs"
RANDOM_SEED = 0


class Training:
    batch_size: int = 128
    cluster_loss_weight: float = 0.8
    dropout_probs: tuple[int, int, int, int] = (0.1, 0.2, 0.3, 0.3)
    early_stopping_patience: int = 7
    minkowski_distance_order: int = 2  # 1: Manhattan distance, 2: Euclidean distance
    learning_rate: float = 1e-3
    lr_plateau_reduction_factor: float = 0.5
    lr_pleateau_reduction_patience: int = 3
    lr_plateau_reduction_min_lr: float = 1e-6
    n_classes: int = 10
    n_data_loader_workers: int = 4
    n_protos_per_class: int = 3
    projection_epochs: list[int] = [15, 50, 100]  # sorted with highest epoch last!
    separation_loss_weight: float = -0.8
