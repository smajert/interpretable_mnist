import math

from matplotlib import pyplot as plt
import numpy as np
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



