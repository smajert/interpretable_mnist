from pathlib import Path
import tempfile

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def load_mnist() -> DataLoader:
    data_transforms = transforms.ToTensor()  # Scales data into [0,1]

    dataset = MNIST(str(tempfile.gettempdir() / Path("MNIST")), transform=data_transforms, download=True)
    return DataLoader(dataset)
