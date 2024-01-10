import lightning.pytorch as pl
import numpy as np
import torch
from torchmetrics import Accuracy

from interpretable_mnist import params
from interpretable_mnist.base_architecture import SimpleConvNetRoot
from interpretable_mnist.core import ProjectedPrototype


def prototype_distances_to_similarities(distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert prototype distances to similarities according to [1], i.e.
    calculating similarity from distance dist via
        sim = log( (dist + 1) / (epsilon +1) )
    with epsilon = 1e-4 and get the global similarity of the
    prototype by subtracting the maximal similarity across the (latent) image
    from the average similarity across the latent image.

    :param distances: [b: n_samples_batch c: n_classes p: n_proto_per_class, h: height, w: width] Prototype distances
        across (latent) image
    :return: [b: n_samples_batch, c: n_classes p: n_proto_per_class] Similarity score of each prototype to the latent,
        [b: n_samples_batch, c: n_classes, p: n_proto_per_class] Minimum distances for each prototype
    """
    min_distances = torch.min(torch.min(distances, dim=-1)[0], dim=-1)[0]  # [b, c, p]
    avg_distances = torch.mean(distances, dim=(-2, -1))  # [b, c, p]

    epsilon = 1e-4
    max_similarities = torch.log((min_distances + 1) / (min_distances + epsilon))  # [b, c, p]
    avg_similarities = torch.log((avg_distances + 1) / (avg_distances + epsilon))  # [b, c, p]
    return max_similarities - avg_similarities, min_distances


def _get_min_in_cluster_distance(labels: torch.Tensor, min_distances: torch.Tensor) -> torch.Tensor:
    """
    Labels must **not** be on-hot encoded but encoded as increasing ints (i.e. 0, 1, 2, 1, 2, ...)

    :param labels: [b] -
    :param min_distances: [b, c, p] -
    :return: [b]
    """
    in_cluster_distances = torch.gather(min_distances, index=labels[:, np.newaxis, np.newaxis], dim=1)  # [b, p]
    in_cluster_min_distances = torch.min(in_cluster_distances, dim=-1)[0].squeeze()  # [b]
    return in_cluster_min_distances


def _get_min_out_cluster_distance(labels: torch.Tensor, min_distances: torch.Tensor) -> torch.Tensor:
    """
    Labels must **not** be on-hot encoded but encoded as increasing ints (i.e. 0, 1, 2, 1, 2, ...)

    :param labels: [b] -
    :param min_distances: [b, c, p] -
    :return:
    """
    in_cluster_mask = torch.full(size=min_distances.shape[0:2], fill_value=False, device=min_distances.device)  # [b, c]
    expanded_labels = labels[:, np.newaxis]
    in_cluster_mask.scatter_(
        dim=1, index=expanded_labels, src=torch.full_like(expanded_labels, fill_value=True, dtype=torch.bool)
    )
    infinities = torch.zeros_like(min_distances)
    infinities[in_cluster_mask, ...] = float("Inf")
    min_distances = min_distances + infinities
    min_distances = torch.flatten(min_distances, start_dim=1)  # [b, c*p]
    out_cluster_min_distances = torch.min(min_distances, dim=-1)[0].squeeze()  # [b]
    return out_cluster_min_distances


# def _get_slot_orthogonality_loss(proto_presence: torch.Tensor) -> torch.Tensor:
#     """
#
#     :param proto_presence: [c, p, s]
#     :return: [scalar]
#     """
#     n_classes = proto_presence.shape[0]
#     n_slots = proto_presence.shape[-1]
#     slot_similarities = torch.nn.functional.cosine_similarity(
#         proto_presence.unsqueeze(2), proto_presence.unsqueeze(-1), dim=1
#     )
#     upper_diagonal_idxs = np.triu_indices(n_slots)
#     slot_similarities[:, upper_diagonal_idxs[0], upper_diagonal_idxs[1]] = 0
#     # todo: cosine similarity is between -1 and 1 -> Unclear whether we should aim for
#     #   it being zero (i.e. sum absolute of slot_similarities) or -1 (i.e. avg_slot_similarities - 1)
#     return torch.abs(slot_similarities).sum() / (n_slots * n_classes)
#     # return slot_similarities.sum() / (n_slots * n_classes) - 1


class ProtoPNetMNIST(pl.LightningModule):
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

        self.minkowski_distance_order = train_info.minkowski_distance_order
        self.learning_rate = train_info.learning_rate
        self.n_classes = train_info.n_classes  # c - Amount of different classes to predict; 10 for MNIST
        self.n_protos_per_class = train_info.n_protos_per_class  # p - Amount of prototypes each class gets

        self.projection_epochs = train_info.projection_epochs
        self.last_epoch = train_info.projection_epochs[-1]
        self.cluster_loss_weight = train_info.cluster_loss_weight
        self.separation_loss_weight = train_info.separation_loss_weight


        # --- Setup convolution layers f: ---
        self.conv_root = SimpleConvNetRoot()  # outputs n_samples_batch x 64 x 3 x 3 -> dim [b, k, h, w]

        # --- Setup prototypes layer g: ---
        prototypes_shape = (self.n_classes, self.n_protos_per_class, prototype_depth, 1, 1)  # [c, p, d, 1, 1]
        self.prototypes = torch.nn.Parameter(torch.rand(prototypes_shape))  # [c, p, d, 1, 1]
        self.projected_prototypes = [[None] * self.n_protos_per_class for _ in range(self.n_classes)]  # [c, p]

        # --- Setup weights that connect only class prototype to prediction for class: ---
        self.output_weights = torch.nn.Parameter(torch.ones((self.n_classes, self.n_protos_per_class)))  # [c, p]
        # note: softmax already in loss function

    def _get_prototype_distances(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculate the minkowski distance of order f between the latent image representation `z`
        and the learned prototypes P, normalized to the prototype depth d.
        In brief:

            return (sum_i |z_i-P_i|**f) ^ (1/f) / d

        :param z: [b: n_samples_batch, k: channels, h: height, w: width] - Latent image representation
        :return: [b: n_samples_batch, c: n_classes, p: n_protos_per_class, h: height, w: width] - (Normalized) distances
            to prototypes
        """
        # prototype shape: [c, p, d, 1, 1]
        f = self.minkowski_distance_order
        z_minus_p = z[:, np.newaxis, np.newaxis, ...] - self.prototypes[np.newaxis, ...]  # [b, c, p, k, h, w]
        return (1 / self.prototypes.shape[2] * torch.sum(torch.abs(z_minus_p) ** f, dim=3)) ** (1/f)   # [b, c, p, h, w]

    def forward(self, x: torch.Tensor):
        z = self.conv_root(x)  # [b, k, h, w]

        prototype_distances = self._get_prototype_distances(z)  # [b, c, p, h, w]
        prototype_similarities, min_distances = prototype_distances_to_similarities(prototype_distances)  # [b, c, p]

        x = torch.sum(prototype_similarities * self.output_weights[np.newaxis, ...], dim=-1)  # [b, c]
        return x, min_distances

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.n_classes).float()
        y_pred, min_distances = self.forward(x)

        if (self.current_epoch == self.projection_epochs[-1]) and (batch_idx == 0):
            self.conv_root.requires_grad_(False)
            self.prototypes.requires_grad_(False)

        if self.current_epoch in self.projection_epochs:
            if batch_idx == 0:  # make sure that no old prototype distances are present
                self.projected_prototypes = [[None] * self.n_protos_per_class for _ in range(self.n_classes)]  # [c, p]
            self.update_projected_prototypes(x)
            return None  # this appears to skip the gradient update, though I am not sure if it is officially supported

        if (self.current_epoch - 1 in self.projection_epochs) and (batch_idx == 0):
            self.push_projected_prototypes()

        self.log(f"weight", self.conv_root.layers[0].weight[0, 0, 0, 0], prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"prototype", self.prototypes[0, 0, 0, 0, 0], prog_bar=True, on_step=False, on_epoch=True)

        entropy_loss = torch.nn.CrossEntropyLoss()(y_pred, y_one_hot)
        cluster_loss = torch.mean(_get_min_in_cluster_distance(y, min_distances))
        separation_loss = torch.mean(_get_min_out_cluster_distance(y, min_distances))


        loss = (
                entropy_loss
                + self.cluster_loss_weight * cluster_loss
                + self.separation_loss_weight * separation_loss
                # + slot_orthogonality_loss
                # todo: l1 loss of last layer
        )

        self.log(f"loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        # self.log(f"cluster_loss", cluster_loss, prog_bar=False, on_step=True, on_epoch=False)
        # self.log(f"separation_loss", separation_loss, prog_bar=False, on_step=True, on_epoch=False)
        # self.log(f"orthogonality_loss", slot_orthogonality_loss, prog_bar=True, on_step=True, on_epoch=True)

        acc_calculator = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
        accuracy = acc_calculator(y_pred, y)
        self.log(f"acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    @torch.no_grad()
    def update_projected_prototypes(
            self,
            batch_data: torch.Tensor,  # [b, K, H, W]
    ) -> None:
        z = self.conv_root(batch_data)  # [b, k, h, w]
        distances_to_protos = torch.sum(torch.abs(
            z[:, np.newaxis, np.newaxis, ...]  # [b, 1, 1, k, h, w]
            - self.prototypes[np.newaxis, ...]  # [1, c, p, d, 1, 1]
        ), dim=3)  # [b, c, p, h, w]

        for c_idx, p_idx in np.ndindex(*self.prototypes.shape[0:2]):  # iterate over indices of each prototype
            distances_to_prototype = distances_to_protos[:, c_idx, p_idx, ...]  # [b, h, w]
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
                best_match_prototype_from_batch.numpy(force=True),
                dist_best_match_to_prototype,
                best_match_training_sample.numpy(force=True),
                best_match_loc,
            )

            if self.projected_prototypes[c_idx][p_idx] is None:
                self.projected_prototypes[c_idx][p_idx] = projected_proto
                continue

            dist_best_match = projected_proto.distance_to_unprojected_prototype
            dist_current = self.projected_prototypes[c_idx][p_idx].distance_to_unprojected_prototype
            if dist_best_match < dist_current:
                self.projected_prototypes[c_idx][p_idx] = projected_proto

    @torch.no_grad()
    def push_projected_prototypes(self):
        proj_prototypes_numpy = np.stack(
            [[proto.prototype for proto in proto_row] for proto_row in self.projected_prototypes]
        )
        proj_prototypes_tensor = torch.from_numpy(proj_prototypes_numpy)[..., np.newaxis, np.newaxis]
        self.prototypes.data = proj_prototypes_tensor.to(self.prototypes.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

