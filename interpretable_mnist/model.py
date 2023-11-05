from interpretable_mnist import params

import lightning.pytorch as pl
import torch


class SimpleConvNet(torch.nn.Module):
    """
    Simple convolutional network to use as a basis for the ProtoPool model.
    """

    def __init__(self, activation: torch.nn.modules.module.Module = torch.nn.ReLU()) -> None:
        super().__init__()
        self.activation = activation

        # fmt: off
        self.layers = torch.nn.ModuleList([                    # input: 1 x 28 x 28
            torch.nn.Conv2d(1, 32, 3, padding="same"),         # 32 x 28 x 28
            self.activation,
            torch.nn.MaxPool2d(2),                             # 32 x 14 x 14
            torch.nn.Conv2d(32, 128, 3, padding="same"),       # 128 x 14 x 14
            self.activation,
            torch.nn.MaxPool2d(2),                             # 128 x 7 x 7
            torch.nn.Conv2d(128, 256, 3, padding="same"),      # 256 x 7 x 7
            self.activation,
            torch.nn.MaxPool2d(2),                             # 256 x 3 x 3
            torch.nn.Conv2d(256, 128, 3, padding="same"),      # 128 x 3 x 3
            self.activation,
            torch.nn.Conv2d(128, 64, 3, padding="same"),       # 64 x 3 x 3
            self.activation,
            torch.nn.Flatten(),                                # 576
            torch.nn.Linear(576, 50),                          # 50
            self.activation,
            torch.nn.Linear(50, 10),                            # 10
        ])
        # fmt: on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ProtoPoolMNIST(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.conv_net = SimpleConvNet()

    def training_step(self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # todo what does this return?
        x, y = batch
        y = torch.nn.functional.one_hot(y, num_classes=10).float()
        conv_out = self.conv_net(x)
        loss = torch.nn.CrossEntropyLoss()(conv_out, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=params.Training.learning_rate)
        return optimizer
