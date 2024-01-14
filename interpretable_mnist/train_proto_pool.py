import torch
import lightning.pytorch as pl

from interpretable_mnist import params
from interpretable_mnist.data import load_mnist
from interpretable_mnist.proto_pnet import ProtoPNetMNIST
from interpretable_mnist.prototype_plot_utilities import plot_projected_prototype


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    mnist_train, mnist_valid = load_mnist(relative_size_split_dataset=0.2)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=params.Training.projection_epochs[-1],
        default_root_dir=params.OUTS_BASE_DIR,
    )
    proto_model = ProtoPNetMNIST(params.Training())

    trainer.fit(
        model=proto_model,
        train_dataloaders=mnist_train,
        val_dataloaders=mnist_valid,
    )

    mnist_test = load_mnist(load_training_data=False, relative_size_split_dataset=0.)
    trainer.test(
        dataloaders=mnist_test,
        ckpt_path="last"
    )

    for class_idx in range(10):
        class_prototypes = proto_model.projected_prototypes[class_idx]
        for proto_idx, class_prototype in enumerate(class_prototypes):
            plot_projected_prototype(
                class_prototype,
                title=(
                    f"class: {class_idx}, prototype: {proto_idx},"
                    f" weight: {proto_model.output_weights[class_idx, proto_idx]:.3f}"
                )
            )


    # plt.figure()
    # plt.pcolormesh(proto_model.output_layer.weight.detach().cpu().numpy())
    # plt.show()
    # plot_projected_prototype(proto_model.projected_prototypes[0])

    # mnist_test = load_mnist(load_training_data=False)
    # trainer.test(
    #     dataloaders=mnist_test,
    #     ckpt_path="last"
    # )