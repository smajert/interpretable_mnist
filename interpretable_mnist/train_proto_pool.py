import torch
import lightning.pytorch as pl

from interpretable_mnist import params
from interpretable_mnist.data import load_mnist
from interpretable_mnist.proto_pool import ProtoPoolMNIST
from interpretable_mnist.proto_pool_plot_utils import plot_projected_prototype


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    mnist_train, mnist_valid = load_mnist(relative_size_split_dataset=0.9)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=params.Training.n_epochs,
        default_root_dir=params.OUTS_BASE_DIR,
    )
    proto_model = ProtoPoolMNIST(params.Training())

    from matplotlib import pyplot as plt
    plt.figure()
    plt.pcolormesh(proto_model.output_layer.weight.detach().cpu().numpy())
    plt.show()

    trainer.fit(
        model=proto_model,
        train_dataloaders=mnist_train,
    )

    plt.figure()
    plt.pcolormesh(proto_model.output_layer.weight.detach().cpu().numpy())
    plt.show()
    plot_projected_prototype(proto_model.projected_prototypes[0])

    # mnist_test = load_mnist(load_training_data=False)
    # trainer.test(
    #     dataloaders=mnist_test,
    #     ckpt_path="last"
    # )