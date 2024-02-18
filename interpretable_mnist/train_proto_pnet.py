import torch
import lightning.pytorch as pl

from interpretable_mnist import params
from interpretable_mnist.data import load_mnist
from interpretable_mnist.proto_pnet import ProtoPNetMNIST
from interpretable_mnist.prototype_plot_utilities import plot_model_prototypes, plot_class_evidence
from interpretable_mnist.train_conv_net import train_conv_net


def train_proto_pnet(do_evaluation: bool) -> ProtoPNetMNIST:
    torch.set_float32_matmul_precision("medium")

    mnist_train, mnist_valid = load_mnist(relative_size_split_dataset=0.2)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=params.Training.projection_epochs[-1] + 1,
        default_root_dir=params.OUTS_BASE_DIR,
    )

    proto_model = ProtoPNetMNIST(params.Training(), len(mnist_train))
    if params.Training.do_transfer_learning:
        conv_net = train_conv_net(do_evaluation=False)
        proto_model.conv_root = conv_net.conv_net_root

    trainer.fit(
        model=proto_model,
        train_dataloaders=mnist_train,
        val_dataloaders=mnist_valid,
    )

    if do_evaluation:
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

        test_batch = next(iter(mnist_test))
        class_evidence = trainer.model.get_evidence_for_class(test_batch[0][1, ...])
        plot_class_evidence(class_evidence)

        augmented_batch = next(iter(mnist_augmented))
        class_evidence = trainer.model.get_evidence_for_class(augmented_batch[0][1, ...])
        plot_class_evidence(class_evidence)

    return trainer.model


if __name__ == "__main__":
    train_proto_pnet(do_evaluation=True)
