import torch


class SimpleConvNetRoot(torch.nn.Module):
    """
    Simple convolutional network to use as a basis for the ProtoPool model.
    """

    def __init__(
        self,
        dropout_probs: tuple[float, float, float, float],
        activation: torch.nn.modules.module.Module = torch.nn.ReLU(),
    ) -> None:
        super().__init__()
        self.activation = activation

        # fmt: off
        self.layers = torch.nn.ModuleList([                     # input: 1 x 28 x 28
            torch.nn.Conv2d(1, 32, 2, stride=2, padding=0),     # 32 x 14 x 14
            torch.nn.BatchNorm2d(32),
            self.activation,
            torch.nn.Dropout(p=dropout_probs[0]),
            torch.nn.Conv2d(32, 128, 2, stride=2, padding=0),   # 128 x 7 x 7
            torch.nn.BatchNorm2d(128),
            self.activation,
            torch.nn.Dropout(p=dropout_probs[1]),
            torch.nn.Conv2d(128, 256, 2, stride=2, padding=0),  # 256 x 3 x 3
            torch.nn.BatchNorm2d(256),
            self.activation,                                    # 256 x 3 x 3
            torch.nn.Dropout(p=dropout_probs[2]),
            torch.nn.Conv2d(256, 128, 1, stride=1, padding=0),  # 128 x 3 x 3
            torch.nn.BatchNorm2d(128),
            self.activation,
            torch.nn.Dropout(p=dropout_probs[3]),
            torch.nn.Conv2d(128, 64, 1, stride=1, padding=0),   # 64 x 3 x 3
            torch.nn.BatchNorm2d(64),
            self.activation,
        ])
        # fmt: on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
