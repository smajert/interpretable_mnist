from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
OUTS_BASE_DIR = REPO_ROOT / "outs"
RANDOM_SEED = 0


@dataclass
class Training:
    """Settings for model training"""

    batch_size: int = 256  # amount of samples in a batch
    constrain_prototypes_to_class: bool = True  # whether to only allow images from a class as prototypes for a class
    dropout_probs: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.3)  # probability settings for dropout layers
    do_batch_norm: bool = False  # whether to use batch norm
    do_transfer_learning: bool = False  # whether to use pre-trained ConvNet as starting point for ProtoPNet
    minkowski_distance_order: int = 2  # 1: Manhattan distance, 2: Euclidean distance
    learning_rate: float = 1e-3
    lr_plateau_reduction_factor: float = 0.5  # factor by which to reduce learning rate when plateau in valid. loss
    lr_plateau_reduction_patience: int = 2  # epochs without improvement before performing learning rate reduction
    lr_plateau_reduction_min_lr: float = 1e-5  # minimum learning rate to reduce to
    n_classes: int = 10  # amount of classes, 10 for MNIST/FashionMNIST
    n_data_loader_workers: int = 8  # amount of workers used when loading the data from disc
    n_protos_per_class: int = 5  # amount of prototypes to use per class
    projection_epochs: tuple[int, ...] = (10, 20, 30, 40)  # when to perform prototypes projection -
    # --- should be sorted with highest epoch last! - Epochs start at 0
    n_freeze_epochs: int = 3  # number of epochs to freeze the convolutional root of the ProtoPNet
    cluster_loss_weight: float = None  # how to weight the cluster loss, `None` means do not use this loss
    l1_loss_weight: float = None  # how to weight the l1-loss for the last layer, `None` means do not use this loss
    orthogonality_loss_weight: float = None  # how to weight the orthogonality loss, `None` means do not use this loss
    separation_loss_weight: float = None  # how to weight the separation loss, `None` means do not use this loss
