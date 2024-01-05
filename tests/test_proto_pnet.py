import math

from matplotlib import pyplot as plt
import numpy as np
import pytest
import torch

from interpretable_mnist import params
from interpretable_mnist import proto_pnet
from interpretable_mnist.data import load_mnist


def test_starting_connections_in_output_layer():
    n_classes = 3
    n_slots = 2
    expected_weights = torch.Tensor([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.]
    ])
    starting_weights = proto_pnet._get_starting_output_layer_class_connections(n_classes, n_slots)

    torch.testing.assert_allclose(starting_weights, expected_weights, rtol=0, atol=1e-11)


def test_prototypes_are_differentiated_when_calling_distance():
    train_config = params.Training()
    train_config.n_classes, train_config.n_prototypes, train_config.n_slots_per_class = 3, 10, 2
    model = proto_pnet.ProtoPNetMNIST(train_config, prototype_depth=64)
    assert model.prototypes.grad is None

    n, k, h, w = 20, 64, 3, 3
    z = torch.rand(n, k, h, w)
    distances = model._get_prototype_distances(z)
    sum_distances = torch.sum(distances)
    sum_distances.backward()
    assert model.prototypes.grad is not None


def original_proto_pool_distance_implementation(x, prototype_vectors, prototype_shape):
    # copied from [1]
    ones = torch.nn.Parameter(torch.ones(prototype_shape), requires_grad=False)
    '''
    apply self.prototype_vectors as l2-convolution filters on input x
    '''
    x2 = x ** 2
    x2_patch_sum = torch.nn.functional.conv2d(input=x2, weight=ones)

    p2 = prototype_vectors ** 2
    p2 = torch.sum(p2, dim=(1, 2, 3))
    # p2 is a vector of shape (num_prototypes,)
    # then we reshape it to (num_prototypes, 1, 1)
    p2_reshape = p2.view(-1, 1, 1)

    xp = torch.nn.functional.conv2d(input=x, weight=prototype_vectors)
    intermediate_result = - 2 * xp + p2_reshape  # use broadcast
    # x2_patch_sum and intermediate_result are of the same shape
    distances = torch.nn.functional.relu(x2_patch_sum + intermediate_result)

    return distances


PROTOTYPES_SHAPE = (10, 1, 64, 1, 1)  # c, p, d, 1, 1


@pytest.mark.parametrize(
    "prototypes", [
        (torch.ones(size=PROTOTYPES_SHAPE)),  # expected result: 0
        (0.5 * torch.ones(size=PROTOTYPES_SHAPE)),  # expected result: (d * (1-0.5))**2 / d
        (np.pi * torch.ones(size=PROTOTYPES_SHAPE)),  # expected result: (d * (1-pi))**2 / d
        # some deviations to [1] for random inputs, but simpler implementation here seems
        # to work better for third example
    ]
)
def test_distance_implementation_equal_to_protopool(prototypes):
    train_config = params.Training()
    train_config.n_classes = PROTOTYPES_SHAPE[0]
    train_config.n_protos_per_class = PROTOTYPES_SHAPE[1]
    model = proto_pnet.ProtoPNetMNIST(train_config, prototype_depth=PROTOTYPES_SHAPE[1])
    model.prototypes = torch.nn.Parameter(prototypes)

    n, k, h, w = 20, PROTOTYPES_SHAPE[2], 3, 3
    z = torch.ones(size=(n, k, h, w))

    distance = model._get_prototype_distances(z)
    pool_proto_shape = (PROTOTYPES_SHAPE[0] * PROTOTYPES_SHAPE[1], *PROTOTYPES_SHAPE[2:])
    prototypes = torch.flatten(prototypes, start_dim=0, end_dim=1)
    distance_original = original_proto_pool_distance_implementation(z, prototypes, pool_proto_shape)

    distance = torch.flatten(distance, start_dim=1, end_dim=2)
    torch.testing.assert_allclose(distance, distance_original, rtol=0, atol=1e-3)


def test_proto_pnet_model_runs():
    train_config = params.Training()
    train_config.n_classes, train_config.n_protos_per_class = 10, 2
    model = proto_pnet.ProtoPNetMNIST(train_config, prototype_depth=64)
    dummy_input = torch.tensor(np.random.uniform(low=0, high=1, size=(10, 1, 28, 28)).astype(np.float32))
    class_pred, _ = model(dummy_input)
    assert class_pred[0].shape[0] == train_config.n_classes


def test_min_in_cluster_distances():
    n_batch = 3
    n_classes = 4
    n_protos_per_class = 2
    min_distances = torch.full(size=(n_batch, n_classes, n_protos_per_class), fill_value=1, dtype=torch.float)
    min_distances[0, 0, 0] = 0.5  # is in "in"-cluster
    min_distances[1, 0, 0] = 0.2  # is in "out"-cluster
    labels = torch.tensor([0, 1, 3], dtype=torch.long)

    min_in_cluster_distances = proto_pnet._get_min_in_cluster_distance(labels, min_distances)
    torch.testing.assert_allclose(min_in_cluster_distances, torch.tensor([0.5, 1.0, 1.0]), rtol=0, atol=1e-11)


def test_min_out_cluster_distances():
    n_batch = 3
    n_classes = 4
    n_protos_per_class = 2
    min_distances = torch.full(size=(n_batch, n_classes, n_protos_per_class), fill_value=1, dtype=torch.float)
    min_distances[0, 0, 0] = 0.5  # is in "in"-cluster
    min_distances[1, 0, 0] = 0.2  # is in "out"-cluster
    labels = torch.tensor([0, 1, 3], dtype=torch.long)

    min_out_cluster_distances = proto_pnet._get_min_out_cluster_distance(labels, min_distances)
    torch.testing.assert_allclose(min_out_cluster_distances, torch.tensor([1.0, 0.2, 1.0]), rtol=0, atol=1e-11)


# def original_proto_pool_dist_loss_implementation(prototype_shape, min_distances, proto_presence, top_k):
#     # copied from [1]
#     #         model, [b, p],        [b, p, n],      [scalar]
#     max_dist = (prototype_shape[1] * prototype_shape[2] * prototype_shape[3])
#     basic_proto = proto_presence.sum(dim=-1).detach()  # [b, p]
#     _, idx = torch.topk(basic_proto, top_k, dim=1)  # [b, n]
#     binarized_top_k = torch.zeros_like(basic_proto)
#     binarized_top_k.scatter_(1, src=torch.ones_like(
#         basic_proto), index=idx)  # [b, p]
#     inverted_distances, _ = torch.max(
#         (max_dist - min_distances) * binarized_top_k, dim=1)  # [b]
#     cost = torch.mean(max_dist - inverted_distances)
#     return cost
#
#
# def test_dist_loss_implementation_equal_to_protopool():
#     n_slots = 2
#     n_prototypes = 3
#     prototype_depth = 64
#     prototype_shape = (n_prototypes, prototype_depth, 1, 1)
#     prototype_size = np.prod(prototype_shape)
#
#     labels = torch.tensor([0, 1, 0])  # n_batch = 3
#     min_dist = torch.tensor([  # n_batch = 3 x n_prototypes = 3
#         [0.2, 0.5, 0.7],
#         [0.1, 0.2, 10],
#         [0.3, 0.4, 0.5],
#     ])
#     proto_presence = torch.tensor([  # n_classes = 2 x n_prototypes = 3 x n_slots = 2
#         [
#             [0, 0], [1, 0], [0, 1]  # prototype 2 and 3 are in slot 1 and 2 of class 1
#         ], [
#             [1, 0], [0, 1], [0, 0]  # prototype 1 and 2 are in slot 1 and 2 of class 2
#         ]
#     ])
#
#     # distance loss for cluster loss:
#     expected_cluster_loss = torch.mean(torch.tensor([0.5, 0.1, 0.4]))
#     cluster_loss = proto_pool._get_distance_loss(
#         labels, min_dist, proto_presence, n_slots_pick=n_slots, prototype_size=prototype_size
#     )
#     cluster_loss_original = original_proto_pool_dist_loss_implementation(
#         prototype_shape, min_dist, proto_presence[labels], n_slots
#     )
#     assert math.isclose(cluster_loss, cluster_loss_original, rel_tol=0, abs_tol=1e-10)
#     assert math.isclose(cluster_loss, expected_cluster_loss, rel_tol=0, abs_tol=1e-10)
#
#     # distance loss for separation loss:
#     expected_separation_loss = torch.mean(torch.tensor([0.2, 10, 0.3]))
#     inverted_proto_presence = 1 - proto_presence
#     separation_loss = proto_pool._get_distance_loss(
#         labels, min_dist, inverted_proto_presence, n_slots_pick=n_prototypes - n_slots, prototype_size=prototype_size
#     )
#     separation_loss_original = original_proto_pool_dist_loss_implementation(
#         prototype_shape, min_dist, inverted_proto_presence[labels], n_prototypes - n_slots
#     )
#     assert math.isclose(separation_loss, separation_loss_original, rel_tol=0, abs_tol=1e-10)
#     assert math.isclose(separation_loss, expected_separation_loss, rel_tol=0, abs_tol=1e-10)
#
#
# def original_proto_pool_orthogonality_loss(proto_presence):
#     # copied from [1]
#     orthogonal_loss_p = torch.nn.functional.cosine_similarity(
#         proto_presence.unsqueeze(2), proto_presence.unsqueeze(-1), dim=1
#     ).sum() / (3 * 10) - 1
#
#     return orthogonal_loss_p
#
#
# def test_orthogonality_loss_runs():
#     proto_presence = torch.rand(size=(10, 30, 3))
#
#     orthogonal_loss = proto_pool._get_slot_orthogonality_loss(proto_presence)
#     # orthongonal_loss_original = original_proto_pool_orthogonality_loss(proto_presence)
#
#     assert orthogonal_loss >= -1
#     assert orthogonal_loss <= 1


def test_prototype_projection():
    train_config = params.Training()
    train_config.n_classes, train_config.n_protos_per_class = 10, 2

    model = proto_pnet.ProtoPNetMNIST(train_config, prototype_depth=64)
    mnist_batch = next(iter(load_mnist()))[0]
    z = model.conv_root(mnist_batch)
    prototype = z[0, :, 0, 0]

    with torch.no_grad():
        model.prototypes[0, 0, ...] = prototype[:, np.newaxis, np.newaxis]
    model.update_projected_prototypes(mnist_batch)
    model.push_projected_prototypes()

    torch.testing.assert_allclose(model.projected_prototypes[0][0].prototype, prototype, rtol=0, atol=1e-11)
    torch.testing.assert_allclose(model.prototypes[0, 0, :, 0, 0], prototype, rtol=0, atol=1e-11)
    assert not torch.allclose(
        torch.from_numpy(model.projected_prototypes[0][1].prototype), prototype, rtol=0, atol=1e-11
    )
    assert not torch.allclose(
        torch.from_numpy(model.projected_prototypes[1][0].prototype), prototype, rtol=0, atol=1e-11
    )
