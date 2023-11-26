import torch
import lightning.pytorch as pl

from interpretable_mnist import params
from interpretable_mnist.data import load_mnist
from interpretable_mnist.proto_pool import ProtoPoolMNIST



if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    mnist_train, mnist_valid = load_mnist(relative_size_split_dataset=0.9)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=params.Training.n_epochs,
        default_root_dir=params.OUTS_BASE_DIR,
    )
    trainer.fit(
        model=ProtoPoolMNIST(params.Training()),
        train_dataloaders=mnist_train,
    )

    # mnist_test = load_mnist(load_training_data=False)
    # trainer.test(
    #     dataloaders=mnist_test,
    #     ckpt_path="last"
    # )