from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectedPrototype:
    prototype: np.ndarray  # [k]
    distance_to_unprojected_prototype: float
    training_sample: np.ndarray  # [K, H, W ]
    prototype_location_in_training_sample: tuple[slice, slice]
