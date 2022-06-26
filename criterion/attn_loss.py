from torch import nn, Tensor
from typing import Tuple


class AttnLoss(nn.Module):
    def __init__(self, pad: int):
        super().__init__()
        self._criterion: nn.Module = nn.CrossEntropyLoss(ignore_index=pad)
        self._pad: int = pad

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        target = target.contiguous().view(-1)
        loss: Tensor = self._criterion(pred.view(-1, pred.size(-1)), target)
        return loss
