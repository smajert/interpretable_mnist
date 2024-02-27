import torch
import lightning.pytorch as pl

from interpretable_mnist import params
from interpretable_mnist.data import load_mnist
from interpretable_mnist.conv_net import ConvNetMNIST


def train_conv_net(do_evaluation: bool) -> ConvNetMNIST:
    torch.set_float32_matmul_precision("medium")

    mnist_train, mnist_valid = load_mnist(relative_size_split_dataset=0.2)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=params.Training.projection_epochs[-1],
        default_root_dir=params.OUTS_BASE_DIR,
    )
    trainer.fit(model=ConvNetMNIST(params.Training()), train_dataloaders=mnist_train, val_dataloaders=mnist_valid)

    if do_evaluation:
        mnist_test = load_mnist(load_training_data=False)
        trainer.test(dataloaders=mnist_test, ckpt_path="last")

        mnist_augmented = load_mnist(load_training_data=False, flip_vertical=True, relative_size_split_dataset=0.0)
        trainer.test(
            dataloaders=mnist_augmented,
            ckpt_path="last",
        )

    return trainer.model


if __name__ == "__main__":
    train_conv_net(do_evaluation=True)
