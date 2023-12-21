import lightning.pytorch as pl
import numpy as np
import torch
from torchmetrics import Accuracy

from interpretable_mnist import params
from interpretable_mnist.base_architecture import SimpleConvNetRoot
from interpretable_mnist.core import ProjectedPrototype


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


def gumbel_cooling_schedule(i_epoch: int, n_cooling_epochs: int) -> float:
    """
    Get tau for the `i_epoch` epoch.

    :param i_epoch: Epoch to get the appropriate tau value for the modified gumbel softmax
    :param n_cooling_epochs: Number of epochs until tau stops decreasing
    :return: tau at epoch `i_epoch`
    """
    start_inv_tau = 1.3
    end_inv_tau = 10 ** 3

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
    prototype_size: int,
) -> torch.Tensor:
    """
    Taken from [1], labels must **not** be one-hot encoded but encoded as increasing ints (i.e. 0, 1, 2, 1, 2, ...)

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

    # We need to add some large, arbitrary number here to all non-associated prototypes so that
    # the zeros for the non-associated prototypes will not be the minimum distance.
    large_number = (1 - binarized_prototype_for_sample) * torch.tensor(2 * prototype_size, requires_grad=False)
    min_distance_to_associated_prototypes = torch.min(
        (min_distances * binarized_prototype_for_sample) + large_number, dim=-1
    )[0]  # [b]

    return torch.mean(min_distance_to_associated_prototypes)


def _get_slot_orthogonality_loss(proto_presence: torch.Tensor) -> torch.Tensor:
    """

    :param proto_presence: [c, p, s]
    :return: [scalar]
    """
    n_classes = proto_presence.shape[0]
    n_slots = proto_presence.shape[-1]
    slot_similarities = torch.nn.functional.cosine_similarity(
        proto_presence.unsqueeze(2), proto_presence.unsqueeze(-1), dim=1
    )
    upper_diagonal_idxs = np.triu_indices(n_slots)
    slot_similarities[:, upper_diagonal_idxs[0], upper_diagonal_idxs[1]] = 0
    # todo: cosine similarity is between -1 and 1 -> Unclear whether we should aim for
    #   it being zero (i.e. sum absolute of slot_similarities) or -1 (i.e. avg_slot_similarities - 1)
    return torch.abs(slot_similarities).sum() / (n_slots * n_classes)
    # return slot_similarities.sum() / (n_slots * n_classes) - 1


class ProtoPoolMNIST(pl.LightningModule):
    def __init__(
        self,
        train_info: params.Training,
        prototype_depth: int = 64
    ) -> None:
        """
        :param prototype_depth: d - Amount of values in each prototype, should be the same as channel amount of
            root convolution layers k
        """
        super().__init__()

        self.learning_rate = train_info.learning_rate
        self.n_classes = train_info.n_classes  # c - Amount of different classes to predict; 10 for MNIST
        self.n_prototypes = train_info.n_prototypes  # p - Amount of prototypes
        self.n_slots = train_info.n_slots_per_class  # s - Amount of prototype slots each class gets
        self.n_cooling_epochs = train_info.n_cooling_epochs
        self.projection_epochs = train_info.projection_epochs
        self.freeze_epoch = train_info.projection_epochs[-1]
        self.cluster_loss_weight = train_info.cluster_loss_weight
        self.separation_loss_weight = train_info.separation_loss_weight
        self.prototype_shape = (self.n_prototypes, prototype_depth, 1, 1)  # [p, d, 1, 1]

        # --- Setup convolution layers f: ---
        self.conv_root = SimpleConvNetRoot()  # outputs n_samples_batch x 64 x 3 x 3 -> dim [b, k, h, w]

        # --- Setup prototypes layer g: ---
        self.proto_presence = torch.nn.Parameter(  # called "q" in [1]
            torch.zeros(self.n_classes, self.n_prototypes, self.n_slots),
        )  # [c, p, s]
        torch.nn.init.xavier_normal_(self.proto_presence, gain=1.0)
        self.prototypes = torch.nn.Parameter(torch.rand(self.prototype_shape))  # [p, d, 1, 1]
        self.projected_prototypes = [None] * self.n_prototypes

        # --- Setup fully connected output layer h: ---
        self.output_layer = torch.nn.Linear(self.n_classes * self.n_slots, self.n_classes, bias=False)
        self.output_layer.weight = torch.nn.Parameter(  # need to transpose to correctly fit into weights of layer
            torch.t(_get_starting_output_layer_class_connections(self.n_classes, self.n_slots))
        )
        self.output_layer.requires_grad_(False)
        # note: softmax already in loss function

    def forward(self, x: torch.Tensor):
        z = self.conv_root(x)  # [b, k, h, w]

        tau = gumbel_cooling_schedule(self.current_epoch, self.n_cooling_epochs)

        proto_presence = modified_gumbel_softmax(self.proto_presence, tau=tau)  # [c, p, s]
        # todo: during prototype projection, the gumbel_softmax should probably be replaced with something
        #       deterministic to get 100% certain prototype assignments (and not have extremely low probability
        #       of different prototype being assigned)

        prototype_distances = self._get_prototype_distances(z)  # [b, p, h, w]
        prototype_similarities, min_distances = prototype_distances_to_similarities(prototype_distances)  # [b, p]
        class_slot_similarities = torch.einsum("bp,cps->bcs", prototype_similarities, proto_presence)  # [b, c, s]
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
        z_minus_p = z[:, np.newaxis, ...] - self.prototypes[np.newaxis, ...]  # [b, p, k, h, w]
        return torch.sum(z_minus_p, dim=2) ** 2 / self.prototypes.shape[1]  # [b, p, h, w]

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.n_classes).float()
        y_pred, min_distances, proto_presence = self.forward(x)

        if (self.current_epoch == self.freeze_epoch) and (batch_idx == 0):
            self.conv_root.requires_grad_(False)
            self.prototypes.requires_grad_(False)
            self.proto_presence.requires_grad_(False)
            self.output_layer.requires_grad_(True)

        if self.current_epoch in self.projection_epochs:
            self.update_projected_prototypes(x)
            return None  # this appears to skip the gradient update, though I am not sure if it is officially supported

        if (self.current_epoch - 1 in self.projection_epochs) and (batch_idx == 0):
            self.push_projected_prototypes()

        self.log(f"weight", self.conv_root.layers[0].weight[0, 0, 0, 0], prog_bar=True, on_step=True, on_epoch=True)

        entropy_loss = torch.nn.CrossEntropyLoss()(y_pred, y_one_hot)

        prototype_size = np.prod(self.prototype_shape)
        cluster_loss = _get_distance_loss(y, min_distances, proto_presence, self.n_slots, prototype_size)

        inverted_proto_presence = 1 - proto_presence
        separation_loss = _get_distance_loss(
            y, min_distances, inverted_proto_presence, self.n_prototypes - self.n_slots, prototype_size
        )

        slot_orthogonality_loss = _get_slot_orthogonality_loss(self.proto_presence)

        loss = (
                entropy_loss
                + self.cluster_loss_weight * cluster_loss
                + self.separation_loss_weight * separation_loss
                # + slot_orthogonality_loss
                # todo: l1 loss of last layer
        )

        self.log(f"loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # self.log(f"cluster_loss", cluster_loss, prog_bar=True, on_step=True, on_epoch=True)
        # self.log(f"separation_loss", separation_loss, prog_bar=True, on_step=True, on_epoch=True)
        # self.log(f"orthogonality_loss", slot_orthogonality_loss, prog_bar=True, on_step=True, on_epoch=True)
        # self.log(f"minimum_proto_presence", torch.min(self.proto_presence), prog_bar=True, on_step=True, on_epoch=True)

        acc_calculator = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
        accuracy = acc_calculator(y_pred, y)
        self.log(f"acc", accuracy, prog_bar=True, on_step=True, on_epoch=True)

        if self.current_epoch >= self.freeze_epoch:
            return entropy_loss
        else:
            return loss

    @torch.no_grad()
    def update_projected_prototypes(
            self,
            batch_data: torch.Tensor,  # [b, K, H, W]
    ) -> None:
        z = self.conv_root(batch_data)  # [b, k, h, w]
        distances_to_protos = torch.sum(torch.abs(
            z[:, np.newaxis, ...]  # [b, 1, k, h, w]
            - self.prototypes[np.newaxis, ...]  # [1, p, d, 1, 1]
        ), dim=2)  # [b, p, h, w]

        for p_idx in range(len(self.projected_prototypes)):
            distances_to_prototype = distances_to_protos[:, p_idx, ...]  # [b, h, w]
            batch_min_idx, height_min_idx, width_min_idx = np.unravel_index(
                torch.argmin(distances_to_prototype).cpu(), distances_to_prototype.shape
            )
            best_match_prototype_from_batch = z[batch_min_idx, :, height_min_idx, width_min_idx]

            dist_best_match_to_prototype = distances_to_prototype[batch_min_idx, height_min_idx, width_min_idx]

            best_match_training_sample = batch_data[batch_min_idx, ...]

            latent_to_input_height = batch_data.shape[2] / z.shape[2]  # = H/h
            proto_start_height = np.rint(latent_to_input_height) * height_min_idx
            proto_stop_height = proto_start_height + latent_to_input_height  # since latent height of prototype is 1
            latent_to_input_width = batch_data.shape[3] / z.shape[3]  # = W/w
            proto_start_width = np.rint(latent_to_input_width) * width_min_idx
            proto_stop_width = proto_start_width + latent_to_input_width  # since latent width of prototype is 1
            best_match_loc = (
                slice(int(proto_start_height), int(proto_stop_height)),
                slice(int(proto_start_width), int(proto_stop_width))
            )

            projected_proto = ProjectedPrototype(
                best_match_prototype_from_batch,
                dist_best_match_to_prototype,
                best_match_training_sample,
                best_match_loc,
            )

            if self.projected_prototypes[p_idx] is None:
                self.projected_prototypes[p_idx] = projected_proto
                continue

            dist_best_match = projected_proto.distance_to_unprojected_prototype
            dist_current = self.projected_prototypes[p_idx].distance_to_unprojected_prototype
            if dist_best_match < dist_current:
                self.projected_prototypes[p_idx] = projected_proto

    @torch.no_grad()
    def push_projected_prototypes(self):
        self.prototypes.data = torch.stack(
            [proj.prototype for proj in self.projected_prototypes], dim=0
        ).to(self.prototypes.device)[..., np.newaxis, np.newaxis]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


