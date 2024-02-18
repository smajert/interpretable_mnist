from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectedPrototype:
    prototype: np.ndarray  # [k]
    distance_to_unprojected_prototype: float
    training_sample: np.ndarray  # [K, H, W ]
    prototype_location_in_training_sample: tuple[slice, slice]


@dataclass
class ClassEvidence:
    projected_prototypes: list[ProjectedPrototype]  # [p]
    sample: np.ndarray  # [H, W]
    prototype_locations_in_sample: list[tuple[slice, slice]]  # [p]
    proto_similarities: np.ndarray  # [p]
    min_distances: np.ndarray  # [p]
    predictions: np.ndarray  # [c]
    proto_weights: np.ndarray  # [p]

