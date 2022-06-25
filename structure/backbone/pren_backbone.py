import torch
from torch import nn, Tensor
from dataset import Alphabet
from .eft_net import eft_builder
from .element import WeightAggregation, GateConv, PoolAggregation
from typing import Tuple


class ATTRBackbone(nn.Module):
    def __init__(self, alphabet: Alphabet, name: str, n_output: int, d_output: int, dropout: float):
        super().__init__()
        self.eft_net: nn.Module = eft_builder(name)
        self.pgg1: nn.Module = PoolAggregation(48, 48, n_output, d_output // 3)
        self.pgg2: nn.Module = PoolAggregation(136, 136, n_output, d_output // 3)
        self.pgg3: nn.Module = PoolAggregation(384, 384, n_output, d_output // 3)
        self.p_gate: nn.Module = GateConv(n_output, alphabet.max_len, d_output, d_output, dropout)

        self.wgg1: nn.Module = WeightAggregation(48, 48, n_output, d_output // 3)
        self.wgg2: nn.Module = WeightAggregation(136, 136, n_output, d_output // 3)
        self.wgg3: nn.Module = WeightAggregation(384, 384, n_output, d_output // 3)
        self.w_gate: nn.Module = GateConv(n_output, alphabet.max_len, d_output, d_output, dropout)
        self.fc: nn.Module = nn.Linear(d_output, alphabet.size())

    def forward(self, image: Tensor) -> Tuple:
        """
            :param image: (B, 3, 32, 128)
            :return: f1, f2, f3 : Là các feature từ eft_net
                    pred :  (B, max_len, d_output)
        """
        f1, f2, f3 = self.eft_net(image)
        rp1: Tensor = self.pgg1(f1)
        rp2: Tensor = self.pgg2(f2)
        rp3: Tensor = self.pgg3(f3)
        rp: Tensor = self.p_gate(torch.cat([rp1, rp2, rp3], dim=2))

        wp1: Tensor = self.wgg1(f1)
        wp2: Tensor = self.wgg2(f2)
        wp3: Tensor = self.wgg3(f3)
        wp: Tensor = self.w_gate(torch.cat([wp1, wp2, wp3], dim=2))
        score: Tensor = (rp + wp) / 2
        pred = self.fc(score)
        return pred
