from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
OUTS_BASE_DIR = REPO_ROOT / "outs"
RANDOM_SEED = 0


class Training:
    batch_size: int = 512
    constrain_prototypes_to_class: bool = True
    dropout_probs: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.3)
    minkowski_distance_order: int = 2  # 1: Manhattan distance, 2: Euclidean distance
    learning_rate: float = 1e-3
    lr_plateau_reduction_factor: float = 0.5
    lr_pleateau_reduction_patience: int = 3
    lr_plateau_reduction_min_lr: float = 1e-5
    n_classes: int = 10
    n_data_loader_workers: int = 8
    n_protos_per_class: int = 5
    projection_epochs: list[int] = [5, 50]  # sorted with highest epoch last!
    cluster_loss_weight: float = None
    l1_loss_weight: float = None
    orthogonality_loss_weight: float = None
    separation_loss_weight: float = None


