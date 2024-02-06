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


def plot_projected_prototype(proto: ProjectedPrototype, axis: plt.Axes) -> None:
    img = proto.training_sample[0, ...]
    height_start = proto.prototype_location_in_training_sample[0].start
    height_stop = proto.prototype_location_in_training_sample[0].stop
    width_start = proto.prototype_location_in_training_sample[1].start
    width_stop = proto.prototype_location_in_training_sample[1].stop

    axis.imshow(img)
    rectangle = patches.Rectangle(
        (width_start, height_start),
        height=height_stop - height_start,
        width=width_stop - width_start,
        linewidth=1,
        edgecolor='r',
        facecolor='none'
    )
    axis.add_patch(rectangle)


def plot_model_prototypes(protos: list[list[ProjectedPrototype]], out_weights: np.ndarray) -> None:
    n_classes = len(protos)
    n_prototypes = len(protos[0])
    fig, axis = plt.subplots(n_prototypes, n_classes)

    for class_idx, class_prototypes in enumerate(protos):
        for proto_idx, proj_proto in enumerate(class_prototypes):
            ax = axis[proto_idx, class_idx]
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False, top=False, labelbottom=False)
            plot_projected_prototype(proj_proto, ax)
            if class_idx == 0:
                ax.set_ylabel(f"Prototype {proto_idx}: ")
            if proto_idx == 0:
                ax.set_title(f"Class {class_idx}: \n Weight: {out_weights[class_idx, proto_idx]:.2f}")
            else:
                ax.set_title(f"Weight: {out_weights[class_idx, proto_idx]:.2f}")
    plt.show()

