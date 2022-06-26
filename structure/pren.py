import time

import torch
import yaml
from torch import nn, Tensor
from dataset import Alphabet
from structure.backbone.eft_net import eft_builder
from structure.backbone.element import WeightAggregation, GateConv, PoolAggregation
from typing import Tuple


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)


class PREN(nn.Module):
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
        self.apply(weight_init)

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


if __name__ == "__main__":
    config_path = r'D:\workspace\project\pren\asset\pc_eb3.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    alphabet = Alphabet(r'D:\workspace\project\pren\asset\viet_alphabet.txt', 256)
    model = PREN(**config['model'], alphabet=alphabet)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params, train_params)
    x = torch.randn((1, 3, 32, 900))
    start = time.time()
    y = model(x)
    print(time.time() - start)
    print(y.size())
