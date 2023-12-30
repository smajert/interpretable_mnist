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
    starting_weights = proto_pool._get_starting_output_layer_class_connections(n_classes, n_slots)

    torch.testing.assert_allclose(starting_weights, expected_weights, rtol=0, atol=1e-11)


def test_visualize_cooling_schedule():
    i_epochs = list(range(50))
    n_cooling_epochs = 30
    tau_schedule = [proto_pool.gumbel_cooling_schedule(i_epoch, n_cooling_epochs) for i_epoch in i_epochs]

    assert math.isclose(tau_schedule[15], 0.00141161833364, rel_tol=0, abs_tol=1e-11)

    do_plot = False
    if do_plot:
        plt.figure()
        plt.plot(i_epochs, tau_schedule)
        plt.xlabel("Epoch")
        plt.ylabel(r"$\tau$")
        plt.gca().set_yscale("log")
        plt.grid(which="both")
        plt.show()


def test_prototypes_are_differentiated_when_calling_distance():
    train_config = params.Training()
    train_config.n_classes, train_config.n_prototypes, train_config.n_slots_per_class = 3, 10, 2
    model = proto_pool.ProtoPoolMNIST(train_config, prototype_depth=64)
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


PROTOTYPES_SHAPE = (10, 64, 1, 1)  # p, d, 1, 1


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
    train_config.n_classes, train_config.n_prototypes, train_config.n_slots_per_class = 3, PROTOTYPES_SHAPE[0], 2
    model = proto_pool.ProtoPoolMNIST(train_config, prototype_depth=PROTOTYPES_SHAPE[1])
    model.prototypes = torch.nn.Parameter(prototypes)

    n, k, h, w = 20, PROTOTYPES_SHAPE[1], 3, 3
    z = torch.ones(size=(n, k, h, w))

    distance = model._get_prototype_distances(z)
    distance_original = original_proto_pool_distance_implementation(z, prototypes, model.prototype_shape)
    torch.testing.assert_allclose(distance, distance_original, rtol=0, atol=1e-3)


def test_proto_pool_model_runs():
    train_config = params.Training()
    train_config.n_classes, train_config.n_prototypes, train_config.n_slots_per_class = 3, 10, 2
    model = proto_pool.ProtoPoolMNIST(train_config, prototype_depth=64)
    dummy_input = torch.tensor(np.random.uniform(low=0, high=1, size=(10, 1, 28, 28)).astype(np.float32))
    class_pred, _, _ = model(dummy_input)
    assert class_pred[0].shape[0] == 3


def original_proto_pool_dist_loss_implementation(prototype_shape, min_distances, proto_presence, top_k):
    # copied from [1]
    #         model, [b, p],        [b, p, n],      [scalar]
    max_dist = (prototype_shape[1] * prototype_shape[2] * prototype_shape[3])
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, p]
    _, idx = torch.topk(basic_proto, top_k, dim=1)  # [b, n]
    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(1, src=torch.ones_like(
        basic_proto), index=idx)  # [b, p]
    inverted_distances, _ = torch.max(
        (max_dist - min_distances) * binarized_top_k, dim=1)  # [b]
    cost = torch.mean(max_dist - inverted_distances)
    return cost


def test_dist_loss_implementation_equal_to_protopool():
    n_slots = 2
    n_prototypes = 3
    prototype_depth = 64
    prototype_shape = (n_prototypes, prototype_depth, 1, 1)
    prototype_size = np.prod(prototype_shape)

    labels = torch.tensor([0, 1, 0])  # n_batch = 3
    min_dist = torch.tensor([  # n_batch = 3 x n_prototypes = 3
        [0.2, 0.5, 0.7],
        [0.1, 0.2, 10],
        [0.3, 0.4, 0.5],
    ])
    proto_presence = torch.tensor([  # n_classes = 2 x n_prototypes = 3 x n_slots = 2
        [
            [0, 0], [1, 0], [0, 1]  # prototype 2 and 3 are in slot 1 and 2 of class 1
        ], [
            [1, 0], [0, 1], [0, 0]  # prototype 1 and 2 are in slot 1 and 2 of class 2
        ]
    ])

    # distance loss for cluster loss:
    expected_cluster_loss = torch.mean(torch.tensor([0.5, 0.1, 0.4]))
    cluster_loss = proto_pool._get_distance_loss(
        labels, min_dist, proto_presence, n_slots_pick=n_slots, prototype_size=prototype_size
    )
    cluster_loss_original = original_proto_pool_dist_loss_implementation(
        prototype_shape, min_dist, proto_presence[labels], n_slots
    )
    assert math.isclose(cluster_loss, cluster_loss_original, rel_tol=0, abs_tol=1e-10)
    assert math.isclose(cluster_loss, expected_cluster_loss, rel_tol=0, abs_tol=1e-10)

    # distance loss for separation loss:
    expected_separation_loss = torch.mean(torch.tensor([0.2, 10, 0.3]))
    inverted_proto_presence = 1 - proto_presence
    separation_loss = proto_pool._get_distance_loss(
        labels, min_dist, inverted_proto_presence, n_slots_pick=n_prototypes - n_slots, prototype_size=prototype_size
    )
    separation_loss_original = original_proto_pool_dist_loss_implementation(
        prototype_shape, min_dist, inverted_proto_presence[labels], n_prototypes - n_slots
    )
    assert math.isclose(separation_loss, separation_loss_original, rel_tol=0, abs_tol=1e-10)
    assert math.isclose(separation_loss, expected_separation_loss, rel_tol=0, abs_tol=1e-10)


def original_proto_pool_orthogonality_loss(proto_presence):
    # copied from [1]
    orthogonal_loss_p = torch.nn.functional.cosine_similarity(
        proto_presence.unsqueeze(2), proto_presence.unsqueeze(-1), dim=1
    ).sum() / (3 * 10) - 1

    return orthogonal_loss_p


def test_orthogonality_loss_runs():
    proto_presence = torch.rand(size=(10, 30, 3))

    orthogonal_loss = proto_pool._get_slot_orthogonality_loss(proto_presence)
    # orthongonal_loss_original = original_proto_pool_orthogonality_loss(proto_presence)
    
    assert orthogonal_loss >= -1
    assert orthogonal_loss <= 1


def test_prototype_projection():
    train_config = params.Training()
    train_config.n_classes, train_config.n_prototypes, train_config.n_slots_per_class = 3, 10, 2

    model = proto_pool.ProtoPoolMNIST(train_config, prototype_depth=64)
    mnist_batch = next(iter(load_mnist()))[0]
    z = model.conv_root(mnist_batch)
    prototype = z[0, :, 0, 0]

    with torch.no_grad():
        model.prototypes[0, ...] = prototype[:, np.newaxis, np.newaxis]
    model.update_projected_prototypes(mnist_batch)
    model.push_projected_prototypes()

    torch.testing.assert_allclose(model.projected_prototypes[0].prototype, prototype, rtol=0, atol=1e-11)
    torch.testing.assert_allclose(model.prototypes[0, :, 0, 0], prototype, rtol=0, atol=1e-11)
