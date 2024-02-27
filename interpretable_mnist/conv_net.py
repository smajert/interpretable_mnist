import lightning.pytorch as pl
import torch
from torchmetrics import Accuracy

from interpretable_mnist import params
from interpretable_mnist.base_architecture import SimpleConvNetRoot


class ConvNetMNIST(pl.LightningModule):
    """
    Simple Convolutional neural Network (ConvNet) to use as a comparison point for
    the ProtoPNet.

    :param train_info: Parameters to use in the model (e.g. learning rate, whether to do batch norm, ....)
    """

    def __init__(
        self,
        train_info: params.Training,
    ) -> None:
        super().__init__()
        self.learning_rate = train_info.learning_rate

        self.conv_net_root = SimpleConvNetRoot(train_info.dropout_probs, train_info.do_batch_norm)  # outputs 64 x 3 x 3
        self.last_layers = torch.nn.Sequential(
            torch.nn.Flatten(),  # 576
            torch.nn.Linear(576, 50),  # 50
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),  # 10
            # note: softmax already in loss
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _generic_batch_processing(self, step_name: str, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_pred = self.last_layers(self.conv_net_root(x))
        loss = torch.nn.CrossEntropyLoss()(y_pred, y_one_hot)
        self.log(f"{step_name}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        acc_calculator = Accuracy(task="multiclass", num_classes=10).to(self.device)
        accuracy = acc_calculator(y_pred, y)
        self.log(f"{step_name}_acc", accuracy, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._generic_batch_processing("train", (batch[0], batch[1]))

    def validation_step(self, val_batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._generic_batch_processing("validation", (val_batch[0], val_batch[1]))

    def test_step(self, test_batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._generic_batch_processing("test", (test_batch[0], test_batch[1]))
