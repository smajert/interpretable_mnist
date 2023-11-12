import numpy as np
import torch

from interpretable_mnist import base_architecture


def test_convnet_root_runs():
    convnet = base_architecture.SimpleConvNetRoot()
    dummy_input = torch.tensor(np.random.uniform(low=0, high=1, size=(10, 1, 28, 28)).astype(np.float32))
    dummy_output = convnet(dummy_input)
    assert dummy_output.shape == torch.Size([10, 64, 3, 3])


