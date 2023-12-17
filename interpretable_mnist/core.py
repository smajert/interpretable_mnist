from dataclasses import dataclass

import torch


@dataclass
class ProjectedPrototype:
    prototype: torch.Tensor  # [k]
    distance_to_unprojected_prototype: float
    training_sample: torch.Tensor  # [K, H, W ]
    prototype_location_in_training_sample: tuple[slice, slice]
