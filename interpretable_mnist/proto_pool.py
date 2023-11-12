from interpretable_mnist import params
from interpretable_mnist.base_architecture import SimpleConvNetRoot

import lightning.pytorch as pl
import torch


def _get_starting_output_layer_class_connections(n_classes: int, n_slots_per_class: int) -> torch.Tensor:
    """
    Get starting connections in fully connected layer. Make positive connections from the
    prototype slots belonging to the class to the class output neuron and negative connections
    from prototype slots of other classes.

    :param n_classes: c - Amount of different classes
    :param n_slots_per_class: s - Amount of slots per class
    :return: [c * s, c] - Weights for properly connected linear layer
    """

    # slots not belonging to class start with -0.5 as per [2]
    starting_weights = torch.full((n_classes * n_slots_per_class, n_classes), fill_value=-0.5)

    for i in range(n_slots_per_class * n_classes):
        # connect first `n_slots_per_class` with the class "0", second `n_slots_per_class` with class "1", etc.
        starting_weights[i, i // n_slots_per_class] = 1

    return starting_weights  # todo write test whether this is correct


class ProtoPoolMNIST(pl.LightningModule):
    def __init__(self, n_classes: int, n_prototypes: int, n_slots_per_class: int, prototype_depth: int) -> None:
        """
        :param n_classes: c - Amount of different classes to predict; 10 different digits for MNIST
        :param n_prototypes: p - Amount of prototypes
        :param n_slots_per_class: s - Amount of prototype slots each class gets
        :param prototype_depth: d - Amount of channels in each prototype
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.n_slots = n_slots_per_class
        self.prototype_shape = (n_prototypes, prototype_depth, 1, 1)  # [p, d, 1, 1]

        # --- Setup convolution layers f: ---
        self.conv_root = SimpleConvNetRoot()  # outputs 64 x 3 x 3

        # --- Setup prototypes layer g: ---
        self.proto_presence = torch.nn.Parameter(  # called "q" in [1]
            torch.zeros(n_classes, n_prototypes, n_slots_per_class)
        )  # [c, p, s]
        torch.nn.init.xavier_normal_(self.proto_presence, gain=1.0)
        self.prototypes = torch.nn.Parameter(torch.rand(self.prototype_shape))  # [p, d, 1, 1]

        # --- Setup fully connected output layer h: ---
        self.output_layer = torch.nn.Linear(n_classes * n_slots_per_class, n_classes, bias=False)
        self.output_layer.requires_gard_ = False
        self.output_layer.weight.data.copy_(  # need to transpose to correctly fit into weights of layer
            torch.t(_get_starting_output_layer_class_connections(n_classes, n_slots_per_class))
        )

    def configure_optimizers(self):
        # todo: This probably needs more optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=params.Training.learning_rate)
        return optimizer


