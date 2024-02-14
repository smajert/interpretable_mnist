import torch
import lightning.pytorch as pl

from interpretable_mnist import params
from interpretable_mnist.data import load_mnist
from interpretable_mnist.proto_pnet import ProtoPNetMNIST
from interpretable_mnist.prototype_plot_utilities import plot_model_prototypes


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    mnist_train, mnist_valid = load_mnist(relative_size_split_dataset=0.2)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=params.Training.projection_epochs[-1] + 1,
        default_root_dir=params.OUTS_BASE_DIR,
    )
    proto_model = ProtoPNetMNIST(params.Training(), len(mnist_train))

    trainer.fit(
        model=proto_model,
        train_dataloaders=mnist_train,
        val_dataloaders=mnist_valid,
    )

    mnist_test = load_mnist(load_training_data=False, relative_size_split_dataset=0.0)
    trainer.test(
        dataloaders=mnist_test,
        ckpt_path="last",
    )

    mnist_augmented = load_mnist(load_training_data=False, do_augmentation=True, relative_size_split_dataset=0.0)
    trainer.test(
        dataloaders=mnist_augmented,
        ckpt_path="last",
    )

    plot_model_prototypes(proto_model.projected_prototypes, proto_model.output_weights.detach().numpy())
