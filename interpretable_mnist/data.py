from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from interpretable_mnist import params


def make_data_loader(dataset: Dataset) -> DataLoader:
    """
    Make a DataLoader from a Dataset with reasonable default settings.

    :param dataset: Dataset to make a DataLoader from
    :return: Dataloader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=params.Training.batch_size,
        pin_memory=True,
        persistent_workers=True,
        num_workers=params.Training.n_data_loader_workers,
    )


def load_mnist(
    load_training_data: bool = True, relative_size_split_dataset: float = 0.0, flip_vertical: bool = False
) -> DataLoader | tuple[DataLoader, DataLoader]:
    """
    Get a `DataLoader` for FashionMNIST using reasonable default values.
    If a split of the data is requested, two `DataLoader`s will be returned, the first containing
    `(1 - relative_size_split_dataset)*100%`  of the data and the second one containing
    `relative_size_split_dataset*100%` of the data.

    :param load_training_data: Whether to load the training or the test data from FashionMNIST.
    :param relative_size_split_dataset: Relative size of the data in the second `DataLoader`. If this
        results in an empty `DataLoader` (e.g. relative_size_split_dataset=0), only the first `DataLoader`
        will be returned.
    :param flip_vertical: Whether to flip the FashionMnist samples vertically
    :return: One or two `DataLoader`s for the data, depending on `relative_size_spit_dataset`
    """
    data_transforms = transforms.ToTensor()  # Scales data into [0,1]
    if flip_vertical:
        data_transforms = transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor()])

    mnist = FashionMNIST(
        str(tempfile.gettempdir() / Path("MNIST")), transform=data_transforms, train=load_training_data, download=True
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

    return make_data_loader(dataset), make_data_loader(split_dataset)
