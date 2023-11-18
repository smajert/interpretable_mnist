import math

from matplotlib import pyplot as plt
import numpy as np
import pytest
import torch

from interpretable_mnist import proto_pool


def test_starting_connections_in_output_layer():
    n_classes = 3
    n_slots = 2
    expected_weights = torch.Tensor([
        [1, -0.5, -0.5],
        [1, -0.5, -0.5],
        [-0.5, 1, -0.5],
        [-0.5, 1, -0.5],
        [-0.5, -0.5, 1],
        [-0.5, -0.5, 1]
    ])
    starting_weights = proto_pool._get_starting_output_layer_class_connections(n_classes, n_slots)

    torch.testing.assert_allclose(starting_weights, expected_weights, rtol=0, atol=1e-11)


def test_visualize_cooling_schedule():
    i_epochs = list(range(50))
    tau_schedule = [proto_pool.gumbel_cooling_schedule(i_epoch) for i_epoch in i_epochs]

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
    c, p, s, d = 3, 10, 2, 64
    model = proto_pool.ProtoPoolMNIST(n_classes=c, n_prototypes=p, n_slots_per_class=s, prototype_depth=d)
    assert model.prototypes.grad is None

    n, C, h, w = 20, 64, 3, 3
    z = torch.rand(n, C, h, w)
    distances = model._prototype_distances(z)
    sum_distances = torch.sum(distances)
    sum_distances.backward()
    assert model.prototypes.grad is not None


def proto_pool_distance_implementation(x, prototype_vectors, prototype_shape):
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
    c, p, s, d = 3, PROTOTYPES_SHAPE[0], 2, PROTOTYPES_SHAPE[1]
    model = proto_pool.ProtoPoolMNIST(n_classes=c, n_prototypes=p, n_slots_per_class=s, prototype_depth=d)
    model.prototypes = torch.nn.Parameter(prototypes)

    n, C, h, w = 20, PROTOTYPES_SHAPE[1], 3, 3
    z = torch.ones(size=(n, C, h, w))

    distance = model._prototype_distances(z)
    distance_protopool = proto_pool_distance_implementation(z, prototypes, model.prototype_shape)
    torch.testing.assert_allclose(distance, distance_protopool, rtol=0, atol=1e-3)





