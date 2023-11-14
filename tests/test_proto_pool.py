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

