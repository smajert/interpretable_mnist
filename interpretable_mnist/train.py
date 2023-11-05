import torch
import lightning.pytorch as pl

from interpretable_mnist import params
from interpretable_mnist.data import load_mnist
from interpretable_mnist.model import ProtoPoolMNIST





if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=params.Training.n_epochs)
    trainer.fit(model=ProtoPoolMNIST(), train_dataloaders=load_mnist())