from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
OUTS_BASE_DIR = REPO_ROOT / "outs"
RANDOM_SEED = 0


class Training:
    batch_size: int = 128
    cluster_loss_weight: float = 0.8
    minkowski_distance_order: int = 2  # 1: Manhattan distance, 2: Euclidean distance
    learning_rate: float = 1e-3
    n_classes: int = 10
    n_data_loader_workers: int = 4
    n_protos_per_class: int = 2
    projection_epochs: list[int] = [50, 80]  # sorted with highest epoch last!
    separation_loss_weight: float = -0.4
