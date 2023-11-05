import numpy as np
import torch

from interpretable_mnist import model


def test_convnet_runs():
    convnet = model.SimpleConvNet()
    dummy_input = torch.tensor(np.random.uniform(low=0, high=1, size=(10, 1, 28, 28)).astype(np.float32))
    dummy_output = convnet(dummy_input)
    print(dummy_output.shape)


