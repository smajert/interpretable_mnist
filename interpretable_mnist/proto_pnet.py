from dataclasses import asdict

import lightning.pytorch as pl
import numpy as np
import torch
from torchmetrics import Accuracy

from interpretable_mnist import params
from interpretable_mnist.base_architecture import SimpleConvNetRoot
from interpretable_mnist.core import ClassEvidence, ProjectedPrototype


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


def _get_prototype_orthogonality_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean absolute cosine similarity between prototypes of the same
    class.

    :param prototypes: [c, p, d, 1, 1] - Prototypes to calculate orthogonality for
    :return: Mean absolute cosine similarity between prototypes of the same class.
    """
    n_classes = prototypes.shape[0]
    n_proto_per_class = prototypes.shape[1]
    # prototypes shape: [c, p, d, 1, 1]
    sum_orthogonality = torch.tensor(0.0)
    for p_idx in range(prototypes.shape[1] - 1):
        rolled_protos = torch.roll(prototypes, shifts=p_idx + 1, dims=1)
        proto_cosine_sim = torch.abs(
            torch.nn.functional.cosine_similarity(prototypes, rolled_protos, dim=2)
        ).squeeze()  # [c, p]
        sum_orthogonality = sum_orthogonality + torch.sum(proto_cosine_sim)

    mean_orthogonality = sum_orthogonality / (n_classes * n_proto_per_class * (n_proto_per_class - 1))
    return mean_orthogonality


def latent_to_input_position(
    width_idx: int, height_idx: int, input_shape: tuple[int, int, int], latent_shape: tuple[int, int, int]
) -> tuple[slice, slice]:
    """
    Convert a position in latent space to the edges of the corresponding rectangle in input/real space.

    Note: This is only valid assuming the map to latent space is done by ConvNets without kernel overlap!
        In other words this is not true in general, i.e. for an arbitrary ConvNet.

    :param width_idx: Width position in latent space
    :param height_idx: Height position in latent space
    :param input_shape: Shape of the input sample (channel x height x width)
    :param latent_shape: Shape of the latent space (channel x height x width)
    :return: Slices marking the rectangular area in input space that corresponds to the given position in latent space
    """
    latent_to_input_height = input_shape[1] / latent_shape[1]  # = H/h
    proto_start_height = np.rint(latent_to_input_height) * height_idx
    proto_stop_height = proto_start_height + latent_to_input_height  # since latent height of prototype is 1
    latent_to_input_width = input_shape[2] / latent_shape[2]  # = W/w
    proto_start_width = np.rint(latent_to_input_width) * width_idx
    proto_stop_width = proto_start_width + latent_to_input_width  # since latent width of prototype is 1
    return (
        slice(int(proto_start_height), int(proto_stop_height)),
        slice(int(proto_start_width), int(proto_stop_width)),
    )


class ProtoPNetMNIST(pl.LightningModule):
    """
    Implementation of a ProtoPNet, similar to what is described in [2].

    :param train_info: Hyperparameters of the model (see interpetable_mnist/params.py for detailed info).
    :param n_training_batches: Amount of batches used during training; this value is important so that the
        learned prototypes can be replaced with prototypes corresponding to parts of the latents of actual
        samples ("pushing prototypes") after the last batch has been processed.
    :param prototype_depth: Amount of entries in a prototype vector; should be the same as channel amount at the
        last root convolutional layer (k)
    """

    def __init__(self, train_info: params.Training, n_training_batches: int, prototype_depth: int = 64) -> None:
        super().__init__()

        self.minkowski_distance_order = train_info.minkowski_distance_order
        self.learning_rate = train_info.learning_rate
        self.lr_plateau_reduction_min_lr = train_info.lr_plateau_reduction_min_lr
        self.lr_plateau_reduction_factor = train_info.lr_plateau_reduction_factor
        self.lr_plateau_reduction_patience = train_info.lr_plateau_reduction_patience

        self.n_classes = train_info.n_classes  # c - Amount of different classes to predict; 10 for MNIST
        self.n_protos_per_class = train_info.n_protos_per_class  # p - Amount of prototypes each class gets
        self.constrain_prototypes_to_class = train_info.constrain_prototypes_to_class

        self.n_freeze_epochs = train_info.n_freeze_epochs
        self.projection_epochs = train_info.projection_epochs
        self.last_epoch = train_info.projection_epochs[-1]

        self.cluster_loss_weight = train_info.cluster_loss_weight
        self.l1_loss_weight = train_info.l1_loss_weight
        self.orthogonality_loss_weight = train_info.orthogonality_loss_weight
        self.separation_loss_weight = train_info.separation_loss_weight

        self.n_training_batches = n_training_batches

        # --- Setup convolution layers f: ---
        self.conv_root = SimpleConvNetRoot(
            train_info.dropout_probs, train_info.do_batch_norm
        )  # outputs n_samples_batch x 64 x 3 x 3 -> dim [b, k, h, w]
        self.conv_root.requires_grad_(False)

        # --- Setup prototypes layer g: ---
        prototypes_shape = (self.n_classes, self.n_protos_per_class, prototype_depth, 1, 1)  # [c, p, d, 1, 1]
        self.prototypes = torch.nn.Parameter(torch.rand(prototypes_shape))  # [c, p, d, 1, 1]
        self.projected_prototypes = [[None] * self.n_protos_per_class for _ in range(self.n_classes)]  # [c, p]

        # --- Setup weights that connect only class prototype to prediction for class: ---
        self.output_weights = torch.nn.Parameter(torch.rand((self.n_classes, self.n_protos_per_class)))  # [c, p]
        # note: softmax already in loss function

        self.save_hyperparameters(asdict(train_info))

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
        return (1 / self.prototypes.shape[2] * torch.sum(torch.abs(z_minus_p) ** f, dim=3)) ** (
            1 / f
        )  # [b, c, p, h, w]

    def forward(self, x: torch.Tensor):
        z = self.conv_root(x)  # [b, k, h, w]

        prototype_distances = self._get_prototype_distances(z)  # [b, c, p, h, w]
        prototype_similarities, min_distances = prototype_distances_to_similarities(prototype_distances)  # [b, c, p]

        x = torch.sum(prototype_similarities * self.output_weights[np.newaxis, ...], dim=-1)  # [b, c]
        return x, min_distances

    # pylint: disable=too-many-branches
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.current_epoch >= self.n_freeze_epochs:
            self.conv_root.requires_grad_(True)

        if (self.current_epoch == self.projection_epochs[-1]) and (batch_idx == 0):
            self.conv_root.requires_grad_(False)
            self.prototypes.requires_grad_(False)
            self.output_weights.requires_grad_(False)
            with torch.no_grad():
                self.output_weights /= torch.sum(self.output_weights)

        x, y = batch

        if self.current_epoch in self.projection_epochs:
            if batch_idx == 0:  # make sure that no old prototype distances are present
                self.projected_prototypes = [[None] * self.n_protos_per_class for _ in range(self.n_classes)]  # [c, p]
                self.optimizers().param_groups[0]["lr"] = self.learning_rate
            self.update_projected_prototypes(x, y)
            if batch_idx == self.n_training_batches - 1:
                print(f"Pushing protos in epoch {self.current_epoch}")
                self.push_projected_prototypes()
            return None  # this appears to skip the gradient update, though I am not sure if it is officially supported

        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.n_classes).float()
        y_pred, min_distances = self.forward(x)

        entropy_loss = torch.nn.CrossEntropyLoss()(y_pred, y_one_hot)
        if self.cluster_loss_weight is not None:
            cluster_loss = torch.mean(_get_min_in_cluster_distance(y, min_distances)) * self.cluster_loss_weight
        else:
            cluster_loss = 0.0
        if self.separation_loss_weight is not None:
            separation_loss = torch.mean(_get_min_out_cluster_distance(y, min_distances)) * self.separation_loss_weight
        else:
            separation_loss = 0.0
        if self.orthogonality_loss_weight is not None:
            orthogonality_loss = _get_prototype_orthogonality_loss(self.prototypes) * self.orthogonality_loss_weight
        else:
            orthogonality_loss = 0.0
        if self.l1_loss_weight is not None:
            l1_loss = (
                torch.linalg.vector_norm(self.output_weights, ord=1) / (self.n_classes * self.n_protos_per_class)
            ) * self.l1_loss_weight
        else:
            l1_loss = 0.0

        loss = entropy_loss + cluster_loss + separation_loss + orthogonality_loss + l1_loss

        acc_calculator = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
        accuracy = acc_calculator(y_pred, y)
        self.log("acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    # pylint: enable=too-many-branches

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int, step_name: str = "val") -> torch.Tensor:
        x, y = batch
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_pred, _ = self.forward(x)
        loss = torch.nn.CrossEntropyLoss()(y_pred, y_one_hot)
        acc_calculator = Accuracy(task="multiclass", num_classes=10).to(self.device)
        accuracy = acc_calculator(y_pred, y)

        self.log(f"{step_name}_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{step_name}_loss", loss)
        return loss

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.validation_step(batch, batch_idx, step_name="test")

    @torch.no_grad()
    def update_projected_prototypes(
        self,
        batch_data: torch.Tensor,  # [b, K, H, W]
        labels: torch.Tensor,  # [b]
    ) -> None:
        z = self.conv_root(batch_data)  # [b, k, h, w]
        distances_to_protos = torch.sum(
            torch.abs(
                z[:, np.newaxis, np.newaxis, ...]  # [b, 1, 1, k, h, w]
                - self.prototypes[np.newaxis, ...]  # [1, c, p, d, 1, 1]
            ),
            dim=3,
        )  # [b, c, p, h, w]

        for c_idx, p_idx in np.ndindex(*self.prototypes.shape[0:2]):  # iterate over indices for each prototype
            distances_to_prototype = distances_to_protos[:, c_idx, p_idx, ...]  # [b, h, w]
            if self.constrain_prototypes_to_class:
                # make sure that only images of the class are used as prototypes for the class
                distances_to_prototype[labels != c_idx, ...] = float("inf")
            batch_min_idx, height_min_idx, width_min_idx = np.unravel_index(
                torch.argmin(distances_to_prototype).cpu(), distances_to_prototype.shape
            )
            best_match_prototype_from_batch = z[batch_min_idx, :, height_min_idx, width_min_idx]

            dist_best_match_to_prototype = distances_to_prototype[batch_min_idx, height_min_idx, width_min_idx]

            best_match_training_sample = batch_data[batch_min_idx, ...]
            best_match_loc = latent_to_input_position(width_min_idx, height_min_idx, batch_data.shape[1:], z.shape[1:])

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
        """Replace learned prototypes with part of latents from the samples."""
        proj_prototypes_numpy = np.stack(
            [[proto.prototype for proto in proto_row] for proto_row in self.projected_prototypes]
        )
        proj_prototypes_tensor = torch.from_numpy(proj_prototypes_numpy)[..., np.newaxis, np.newaxis]
        self.prototypes.data = proj_prototypes_tensor.to(self.prototypes.device)

    def get_evidence_for_class(self, single_image: torch.Tensor, class_idx: None | int = None) -> ClassEvidence:
        """
        Detailed classification of a single image/sample.

        :param single_image: Image to classify
        :param class_idx: Index of the class to get the prototypes for; if `None`, the class that `single_image`
            is predicted as will be chosen.
        :return: Information and results of the classification, including information about the prototypes for the
            requested `class_idx`.
        """
        if self.projected_prototypes[0][0] is None:
            raise RuntimeError("No projected prototypes yet, cannot make detailed prediction.")
        batch = single_image[np.newaxis, ...]
        z = self.conv_root(batch)  # [1, k, h, w]
        prototype_distances = self._get_prototype_distances(z)  # [1, c, p, h, w]
        prototype_similarities, min_distances = prototype_distances_to_similarities(prototype_distances)  # [1, c, p]
        prediction = torch.sum(prototype_similarities * self.output_weights[np.newaxis, ...], dim=-1)  # [1, c]
        if class_idx is None:
            class_idx = torch.argmax(prediction)

        proto_best_match_locations = []
        for p_idx in range(self.prototypes.shape[1]):
            distances_to_prototype = prototype_distances[:, class_idx, p_idx, ...]  # [1, h, w]
            _, height_min_idx, width_min_idx = np.unravel_index(
                torch.argmin(distances_to_prototype).cpu(), distances_to_prototype.shape
            )
            proto_best_match_locations.append(
                latent_to_input_position(width_min_idx, height_min_idx, batch.shape[1:], z.shape[1:])
            )
        return ClassEvidence(
            self.projected_prototypes[class_idx],
            single_image[0, ...].detach().numpy(),
            proto_best_match_locations,
            prototype_similarities[0, class_idx, :],
            min_distances[0, class_idx, :],
            prediction[0, :].detach().numpy(),
            self.output_weights[class_idx, :].detach().numpy(),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_plateau_reduction_factor,
                patience=self.lr_plateau_reduction_patience,
                min_lr=self.lr_plateau_reduction_min_lr,
                verbose=True,
            ),
            "monitor": "val_loss",  # the quantity to be monitored
        }
        return [optimizer], [scheduler]
