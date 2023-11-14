from interpretable_mnist import params
from interpretable_mnist.base_architecture import SimpleConvNetRoot

import lightning.pytorch as pl
import numpy as np
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

    return starting_weights


def modified_gumbel_softmax(proto_presence: torch.Tensor, tau: float) -> torch.Tensor:
    r"""
    Gumbel softmax with modified scaling according to [1], section 8 (supplement).
    Different from the normal Gumbel softmax, this version starts to consider noise
    (i.e. the values drawn from the Gumbel distribution $\eta_i$) less as tau decreases.

    :param proto_presence: [c, p, s]: Tensor assigning prototypes to classes (q in [1])
    :param tau: non-negative scalar temperature
    :return: [c, p, s] - Modified Gumbel softmax applied to `proto_presence`
    """
    return torch.nn.functional.gumbel_softmax(proto_presence / tau, tau=1, dim=1)


def gumbel_cooling_schedule(i_epoch: int) -> float:
    """
    Get tau for the `i_epoch` epoch.

    :param i_epoch: Epoch to get the appropriate tau value for the modified gumbel softmax
    :return: tau at epoch `i_epoch`
    """
    start_inv_tau = 1.3
    end_inv_tau = 10 ** 3
    n_cooling_epochs = 30

    # alpha from [1]: alpha = (end_inv_tau / start_inv_tau) ** 2 / epoch_interval
    inv_tau = (
        end_inv_tau * np.sqrt(i_epoch / n_cooling_epochs) + start_inv_tau
        if i_epoch < n_cooling_epochs else end_inv_tau
    )
    return 1 / inv_tau


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
        self.output_layer.requires_grad_ = False
        self.output_layer.weight.data.copy_(  # need to transpose to correctly fit into weights of layer
            torch.t(_get_starting_output_layer_class_connections(n_classes, n_slots_per_class))
        )

    def forward(self, x: torch.Tensor):
        tau = gumbel_cooling_schedule(self.current_epoch)
        proto_presence = modified_gumbel_softmax(self.proto_presence, tau=tau)
        # todo: implement gumbel scaling as described in [1] supplement 8
        pass


    def configure_optimizers(self):
        # todo: This probably needs more optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=params.Training.learning_rate)
        return optimizer


