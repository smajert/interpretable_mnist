from itertools import cycle

from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from prettytable import PrettyTable

from interpretable_mnist.core import ClassEvidence, ProjectedPrototype

FASHION_MNIST_CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def _plot_projected_prototype(proto: ProjectedPrototype, axis: plt.Axes, rect_color: str = "r") -> None:
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
        linewidth=3,
        edgecolor=rect_color,
        facecolor="none",
    )
    axis.add_patch(rectangle)


def _make_axis_ticks_invisible_inplace(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)


def plot_model_prototypes(protos: list[list[ProjectedPrototype]], out_weights: np.ndarray) -> None:
    """
    Plot all prototypes that are part of the model to get a sense as to what the model deems
    important in the data. The prototypes themselves are marked by red rectangles in the
    image of the sample they are taken from.

    :param protos: [n_classes, n_prototypes] - Projected prototypes from a trained ProtoPnet
    :param out_weights: [n_classes, n_prototypes] - Weights for each prototypes
    """
    n_classes = len(protos)
    n_prototypes = len(protos[0])
    _, axis = plt.subplots(n_prototypes, n_classes)

    for class_idx, class_prototypes in enumerate(protos):
        for proto_idx, proj_proto in enumerate(class_prototypes):
            ax = axis[proto_idx, class_idx]
            _make_axis_ticks_invisible_inplace(ax)
            _plot_projected_prototype(proj_proto, ax)
            if class_idx == 0:
                ax.set_ylabel(f"Prototype {proto_idx}: ")
            if proto_idx == 0:
                ax.set_title(
                    f"Class {FASHION_MNIST_CLASS_NAMES[class_idx]}: \n Weight: {out_weights[class_idx, proto_idx]:.2f}"
                )
            else:
                ax.set_title(f"Weight: {out_weights[class_idx, proto_idx]:.2f}")
    plt.show()


def plot_class_evidence(class_evidence: ClassEvidence) -> None:
    """
    Make a nice plot of the prediction for a single sample, where for each prototype of the
    predicted class a differently colored rectangle is drawn.

    :param class_evidence: Prediction results as returned by the `get_evidence_for_class`-function from a
        `ProtoPNetMNIST` object
    """
    table = PrettyTable()
    table.field_names = FASHION_MNIST_CLASS_NAMES
    table.add_row([f"{pred:.3f}" for pred in class_evidence.predictions.tolist()])

    n_prototypes = len(class_evidence.projected_prototypes)
    property_cycler = cycle(list(zip(["r", "g", "b", "y", "c", "m", "w", "k"], [8, 7, 6, 5, 4, 3, 2, 1])))
    proto_props = [next(iter(property_cycler)) for _ in range(n_prototypes)]

    plt.figure()
    sample_axis = plt.gca()
    plt.imshow(class_evidence.sample)
    _make_axis_ticks_invisible_inplace(sample_axis)
    plt.title(table.get_string(), fontname="monospace")
    for proto_idx, rect_props in enumerate(proto_props):
        height_start = class_evidence.prototype_locations_in_sample[proto_idx][0].start
        height_stop = class_evidence.prototype_locations_in_sample[proto_idx][0].stop
        width_start = class_evidence.prototype_locations_in_sample[proto_idx][1].start
        width_stop = class_evidence.prototype_locations_in_sample[proto_idx][1].stop
        rectangle = patches.Rectangle(
            (width_start, height_start),
            height=height_stop - height_start,
            width=width_stop - width_start,
            linewidth=rect_props[1],
            edgecolor=rect_props[0],
            facecolor="none",
        )
        sample_axis.add_patch(rectangle)

    _, proto_axis = plt.subplots(n_prototypes, 2)
    class_prototypes = class_evidence.projected_prototypes
    for proto_idx, proj_proto in enumerate(class_prototypes):
        ax = proto_axis[proto_idx, 0]
        _make_axis_ticks_invisible_inplace(ax)
        _plot_projected_prototype(proj_proto, ax, rect_color=proto_props[proto_idx][0])
        text_ax = proto_axis[proto_idx, 1]
        _make_axis_ticks_invisible_inplace(text_ax)
        text_ax.text(
            0.5,
            0.5,
            (
                f"Similarity: {class_evidence.proto_similarities[proto_idx]:.2f}\n"
                f" Weight: {class_evidence.proto_weights[proto_idx]:.4f}"
            ),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=24,
            transform=text_ax.transAxes,
        )

    plt.show()
