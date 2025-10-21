from typing import override

from torch import Tensor
from torch.nn import Module


class Encoder(Module):
    def __init__(self):
        super().__init__()

    def encode(self, x: Tensor) -> Tensor:  # pyright: ignore[reportReturnType]
        pass

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)


class MLPEncoder(Encoder):
    def __init__(self, mlp: Module):
        super().__init__()
        self.mlp = mlp

    @override
    def encode(self, x: Tensor) -> Tensor:
        return self.mlp(x)
