from interpretable_mnist import params
from interpretable_mnist.base_architecture import SimpleConvNetRoot

import lightning.pytorch as pl
import numpy as np
import torch
from torchmetrics import Accuracy


def _get_starting_output_layer_class_connections(n_classes: int, n_slots_per_class: int) -> torch.Tensor:
    """
    Get starting connections in fully connected layer. Make positive connections from the
    prototype slots belonging to the class to the class output neuron and negative connections
    from prototype slots of other classes.

    :param n_classes: c - Amount of different classes
    :param n_slots_per_class: s - Amount of slots per class
    :return: [c * s, c] - Weights for properly connected linear layer
    """

    # slots not belonging to class are not set to -0.5 as per [2] because similarities to
    # prototypes of other classes should not count as evidence against the class if prototypes are shared
    starting_weights = torch.full((n_classes * n_slots_per_class, n_classes), fill_value=0.)

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


def prototype_distances_to_similarities(distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert prototype distances to similarities according to [1], i.e.
    calculating similarity from distance dist via
        sim = log( (dist + 1) / (epsilon +1) )
    with epsilon = 1e-4 and get the global similarity of the
    prototype by subtracting the maximal similarity across the (latent) image
    from the average similarity across the latent image.

    :param distances: [b: n_samples_batch, p: n_prototypes, h: height, w: width] Prototype distances
        across (latent) image
    :return: [b: n_samples_batch, p: n_prototypes] Similarity score of each prototype to the (latent) image,
        [b: n_samples_batch, p: n_prototypes] Minimum distances for each prototype
    """
    min_distances = torch.min(torch.min(distances, dim=-1)[0], dim=-1)[0]  # [b, p]
    avg_distances = torch.mean(distances, dim=(-2, -1))  # [b, p]

    epsilon = 1e-4
    max_similarities = torch.log((min_distances + 1) / (min_distances + epsilon))  # [b, p]
    avg_similarities = torch.log((avg_distances + 1) / (avg_distances + epsilon))  # [b, p]
    return max_similarities - avg_similarities, min_distances


def _get_distance_loss(
    labels: torch.Tensor,
    min_distances: torch.Tensor,
    proto_presence: torch.Tensor,
    n_slots_pick: int,
) -> torch.Tensor:
    """
    Taken from [1], labels must **not** be one-hot encoded but encoded as increasing ints (i.e. 0, 1, 2, 1, 2, ...)

    todo: subtraction from max possible distance in [1, 2], check if this is necessary

    :param labels: [b] -
    :param min_distances: [b, p]
    :param proto_presence: [c, p, s]
    :param n_slots_pick: S - Amount of prototypes to pick from `proto_presence`
    :return: [scalar]
    """
    relevant_prototypes_for_sample = proto_presence[labels, ...]  # [b, p, s]
    # should be high for prototypes associated with a class of a given sample and low for other prototypes:
    prototypes_for_sample = torch.sum(relevant_prototypes_for_sample, dim=-1).detach()  # [b, p]

    _, idx = torch.topk(prototypes_for_sample, n_slots_pick, dim=1)  # [b, S] - indices to prototypes in slots

    binarized_prototype_for_sample = torch.zeros_like(prototypes_for_sample)  # [b, p]; 1 where proto associated else 0
    binarized_prototype_for_sample.scatter_(dim=1, src=torch.ones_like(prototypes_for_sample), index=idx)  # [b, p]

    min_distance_to_associated_prototypes = torch.min(min_distances * binarized_prototype_for_sample)  # [b]

    return torch.mean(min_distance_to_associated_prototypes)

class ProtoPoolMNIST(pl.LightningModule):
    def __init__(
        self,
        train_info: params.Training,
        prototype_depth: int = 64
    ) -> None:
        """
        :param prototype_depth: d - Amount of values in each prototype, should be the same as channel amount of
            root convolution layers C
        """
        super().__init__()

        self.n_classes = train_info.n_classes  # c - Amount of different classes to predict; 10 for MNIST
        self.n_prototypes = train_info.n_prototypes  # p - Amount of prototypes
        self.n_slots = train_info.n_slots_per_class  # s - Amount of prototype slots each class gets
        self.config = train_info
        self.prototype_shape = (self.n_prototypes, prototype_depth, 1, 1)  # [p, d, 1, 1]

        # --- Setup convolution layers f: ---
        self.conv_root = SimpleConvNetRoot()  # outputs n_samples_batch x 64 x 3 x 3 -> dim [b, C, h, w]

        # --- Setup prototypes layer g: ---
        self.proto_presence = torch.nn.Parameter(  # called "q" in [1]
            torch.zeros(self.n_classes, self.n_prototypes, self.n_slots)
        )  # [c, p, s]
        torch.nn.init.xavier_normal_(self.proto_presence, gain=1.0)
        self.prototypes = torch.nn.Parameter(torch.rand(self.prototype_shape))  # [p, d, 1, 1]

        # --- Setup fully connected output layer h: ---
        self.output_layer = torch.nn.Linear(self.n_classes * self.n_slots, self.n_classes, bias=False)
        self.output_layer.requires_grad_ = False
        self.output_layer.weight.data.copy_(  # need to transpose to correctly fit into weights of layer
            torch.t(_get_starting_output_layer_class_connections(self.n_classes, self.n_slots))
        )
        # note: softmax already in loss function

    def forward(self, x: torch.Tensor):
        z = self.conv_root(x)  # [b, C, h, w]

        tau = gumbel_cooling_schedule(self.current_epoch)
        proto_presence = modified_gumbel_softmax(self.proto_presence, tau=tau)  # [c, p, s]

        prototype_distances = self._get_prototype_distances(z)  # [b, p, h, w]
        prototype_similarities, min_distances = prototype_distances_to_similarities(prototype_distances)  # [b, p]
        class_slot_similarities = torch.einsum("np,cps->ncs", prototype_similarities, proto_presence)  # [b, c, s]
        class_slot_similarities = torch.flatten(class_slot_similarities, start_dim=1)  # [b, c*s]

        x = self.output_layer(class_slot_similarities)  # [b, c]
        return x, min_distances, proto_presence

    def _get_prototype_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate distance between the latent image representation `z`
        and the learned prototypes p, normalized to the prototype depth d,
        which should be equal to the amount of channels C.
        In brief:

            return |z-p|**2 / d

        :param z: [b: n_samples_batch, C: Channels, h: height, w: width] - Latent image representation
        :return: [b: n_samples_batch, p: n_prototypes, h: height, w: width] - (Normalized) distances to prototypes
        """
        # prototype shape: [p, d, 1, 1]
        z_minus_p = z[:, np.newaxis, ...] - self.prototypes[np.newaxis, ...]  # [b, p, C, h, w]
        return torch.abs(torch.sum(z_minus_p, dim=2)) ** 2 / self.prototypes.shape[1]  # [b, p, h, w]

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_pred, min_distances, proto_presence = self.forward(x)

        entropy_loss = torch.nn.CrossEntropyLoss()(y_pred, y_one_hot)
        cluster_loss = _get_distance_loss(y, min_distances, proto_presence, self.n_slots)
        inverted_proto_presence = 1 - proto_presence
        separation_loss = _get_distance_loss(
            y, min_distances, inverted_proto_presence, self.n_prototypes - self.n_slots
        )

        loss = (
                entropy_loss
                + self.config.cluster_loss_weight * cluster_loss
                + self.config.separation_loss_weight * separation_loss
        )

        self.log(f"loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        acc_calculator = Accuracy(task="multiclass", num_classes=10).to(self.device)
        accuracy = acc_calculator(y_pred, y)
        self.log(f"acc", accuracy, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # todo: This probably needs more optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer


