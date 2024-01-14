from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from interpretable_mnist import params


def make_data_loader(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=params.Training.batch_size,
        pin_memory=True,
        persistent_workers=True,
        num_workers=params.Training.n_data_loader_workers
    )


def load_mnist(
    load_training_data: bool = True, relative_size_split_dataset: float = 0.
) -> DataLoader | tuple[DataLoader, DataLoader]:
    data_transforms = transforms.ToTensor()  # Scales data into [0,1]

    mnist = MNIST(
        str(tempfile.gettempdir() / Path("MNIST")),
        transform=data_transforms,
        train=load_training_data,
        download=True
    )
    seed = torch.Generator().manual_seed(params.RANDOM_SEED)
    split_dataset_size = int(relative_size_split_dataset * len(mnist))
    dataset_size = len(mnist) - split_dataset_size
    if split_dataset_size >= 1:
        dataset, split_dataset = random_split(mnist, [dataset_size, split_dataset_size], generator=seed)
    else:
        dataset, split_dataset = mnist, None

    if split_dataset is None:
        return make_data_loader(dataset)
    else:
        return make_data_loader(dataset), make_data_loader(split_dataset)



