from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectedPrototype:
    """
    Holds information about a single projected prototype for the ProtoPNet model.

    :param prototype: [d] - Prototype consisting of d == k entries, where k is the amount of latent channels
    :param distance_to_unprojected_prototype: Distance to the unprojected prototype (i.e. the 'messy' prototype
        before projection that did not actually correspond to part of a training image).
    :param training_sample: [K, H, W] - Training sample the prototype is from, K == 1 for grayscale images
    :param prototype_location_in_training_sample: Part of the training sample in height- and width-direction that
        the prototypes corresponds to
    """

    prototype: np.ndarray  # [k]
    distance_to_unprojected_prototype: float
    training_sample: np.ndarray  # [K, H, W ]
    prototype_location_in_training_sample: tuple[slice, slice]


@dataclass
class ClassEvidence:
    """
    Result of a detailed prediction performed with the ProtoPNet model.

    :param projected_prototypes: [p] - All prototypes of the model belonging to the predicted class
    :param sample: [H, W] - sample to predict, assuming a grayscale image (K == 1)
    :param prototype_locations_in_sample: [p] - Position of the highest similarity within the sample for
        each prototype of the predicted class
    :param proto_similarities: [p] - Highest similarity for each prototype of the predicted class
    :param min_distances: [p] - Minimum distance to the sample for each prototype of the predicted class
    :param predictions: [c] - Predictions for every class
    :param proto_weights: [p] - Weight of the output layer of the ProtoPNet for each prototype of the
        predicted class
    """

    projected_prototypes: list[ProjectedPrototype]  # [p]
    sample: np.ndarray  # [H, W]
    prototype_locations_in_sample: list[tuple[slice, slice]]  # [p]
    proto_similarities: np.ndarray  # [p]
    min_distances: np.ndarray  # [p]
    predictions: np.ndarray  # [c]
    proto_weights: np.ndarray  # [p]
