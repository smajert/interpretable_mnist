from interpretable_mnist import params

import lightning.pytorch as pl
import torch
from torchmetrics import Accuracy


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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=params.Training.learning_rate)
        return optimizer

    def generic_batch_processing(
        self, step_name: str, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        x, y = batch
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_pred = self.conv_net(x)
        loss = torch.nn.CrossEntropyLoss()(y_pred, y_one_hot)
        self.log(f"{step_name}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        acc_calculator = Accuracy(task="multiclass", num_classes=10).to(self.device)
        accuracy = acc_calculator(y_pred, y)
        self.log(f"{step_name}_acc", accuracy, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.generic_batch_processing("train", (batch[0], batch[1]))

    def validation_step(self, val_batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.generic_batch_processing("validation", (val_batch[0], val_batch[1]))

    def test_step(self, test_batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.generic_batch_processing("test", (test_batch[0], test_batch[1]))
