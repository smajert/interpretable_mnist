from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
OUTS_BASE_DIR = REPO_ROOT / "outs"
RANDOM_SEED = 0


@dataclass
class Training:
    batch_size: int = 256
    constrain_prototypes_to_class: bool = True
    dropout_probs: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.3)
    do_batch_norm: bool = False
    do_transfer_learning: bool = True
    minkowski_distance_order: int = 2  # 1: Manhattan distance, 2: Euclidean distance
    learning_rate: float = 1e-3
    lr_plateau_reduction_factor: float = 0.5
    lr_pleateau_reduction_patience: int = 3
    lr_plateau_reduction_min_lr: float = 1e-5
    n_classes: int = 10
    n_data_loader_workers: int = 8
    n_protos_per_class: int = 5
    projection_epochs: tuple[int, ...] = (5, 10, 20)  # sorted with highest epoch last! - Epochs start at 0
    n_freeze_epochs: int = 2
    cluster_loss_weight: float = None
    l1_loss_weight: float = None
    orthogonality_loss_weight: float = None
    separation_loss_weight: float = None


