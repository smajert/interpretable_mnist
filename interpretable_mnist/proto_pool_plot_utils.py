from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_prototype_presence(proto_presence: torch.Tensor, class_idx: int, slot_idx: int) -> None:
    proto_presence_numpy = proto_presence.numpy()
    prototype_axis = np.arange(0, len(proto_presence_numpy[class_idx, :, slot_idx]))

    plt.figure()
    plt.title(f"Prototype assignment for slot {slot_idx} of class {class_idx}")
    plt.bar(prototype_axis, proto_presence_numpy)
    plt.xlabel("Prototype index")
    plt.ylabel("Assignment")
    plt.grid()
    plt.show()
