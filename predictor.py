import numpy as np
import torch
from torch import nn, Tensor
from dataset import Alphabet, resize, normalize
from typing import Dict
from structure import PREN
import yaml


class PRENPredictor:
    def __init__(self, config: str, pretrained: str):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        with open(config) as f:
            data: Dict = yaml.safe_load(f)
        self.alphabet: Alphabet = Alphabet(**data['alphabet'])
        self.model: nn.Module = PREN(self.alphabet, **data['model'])
        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        state_dict = torch.load(pretrained, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])

    def predict(self, image: np.ndarray) -> str:
        self.model.eval()
        with torch.no_grad():
            image = resize(image, [32, 900], 0)
            image = normalize(image)
            input: Tensor = torch.from_numpy(image).unsqueeze(0)
            input = input.permute(0, 3, 1, 2).to(self.device)
            pred = self.model(input.float())
            rp: np.ndarray = pred.softmax(dim=2).cpu().detach().numpy().argmax(axis=2)
            p_str: str = self.alphabet.decode(rp[0])
            return p_str
