from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import torch

from interpretable_mnist.core import ProjectedPrototype


def plot_prototype_presence(proto_presence: torch.Tensor, class_idx: int, slot_idx: int) -> None:
    proto_presence_numpy = proto_presence.detach().cpu().numpy()
    prototype_axis = np.arange(0, len(proto_presence_numpy[class_idx, :, slot_idx]))

    plt.figure()
    plt.title(f"Prototype assignment for slot {slot_idx} of class {class_idx}")
    plt.bar(prototype_axis, proto_presence_numpy[class_idx, :, slot_idx])
    plt.xlabel("Prototype index")
    plt.ylabel("Assignment")
    plt.grid()
    plt.show()


def plot_projected_prototype(proto: ProjectedPrototype, title: str | None = None) -> None:
    img = proto.training_sample[0, ...]
    height_start = proto.prototype_location_in_training_sample[0].start
    height_stop = proto.prototype_location_in_training_sample[0].stop
    width_start = proto.prototype_location_in_training_sample[1].start
    width_stop = proto.prototype_location_in_training_sample[1].stop

    plt.figure()
    plt.imshow(img)
    plt.title(title)

    rectangle = patches.Rectangle(
        (width_start, height_start),
        height=height_stop - height_start,
        width=width_stop - width_start,
        linewidth=1,
        edgecolor='r',
        facecolor='none'
    )
    plt.gca().add_patch(rectangle)
    plt.show()
