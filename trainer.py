import numpy as np
import torch
import Levenshtein
from torch import nn, optim, Tensor
from structure import PRENModel
from dataset import ATTRLoader, Alphabet
from typing import Dict, Tuple
from utilities.averager import Averager
from utilities import PRENCheckpoint, PRENLogger
from criterion import AttnLoss


class PRENTrainer:
    def __init__(self,
                 model: Dict,
                 alphabet: Dict,
                 optimizer: Dict,
                 total_epoch: int,
                 save_interval: int,
                 start_epoch: int,
                 clip_grad_norm: float,
                 train: Dict,
                 valid: Dict,
                 test: Dict,
                 checkpoint: Dict,
                 logger: Dict):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.total_epoch: int = total_epoch
        self.save_interval: int = save_interval
        self.start_epoch: int = start_epoch
        self.clip_grad_norm: float = clip_grad_norm

        self.alphabet: alphabet = Alphabet(**alphabet)
        self.model: nn.Module = PRENModel(**model, alphabet=self.alphabet)
        self.model = self.model.to(self.device)
        self.criterion: nn.Module = AttnLoss(self.alphabet.pad)
        self.criterion = self.criterion.to(self.device)
        cls = getattr(optim, optimizer['name'])
        self.optimizer: optim.Optimizer = cls(self.model.parameters(), **optimizer['params'])
        self.train_loader = ATTRLoader(**train, alphabet=self.alphabet).build()
        self.valid_loader = ATTRLoader(**valid, alphabet=self.alphabet).build()
        self.test_loader = ATTRLoader(**test, alphabet=self.alphabet).build()

        self.logger: PRENLogger = PRENLogger(**logger)
        self.checkpoint: PRENCheckpoint = PRENCheckpoint(**checkpoint)
        self.step: int = 0

    def train(self):
        self.load()
        self.logger.partition_report()
        self.logger.time_report("Starting:")
        self.logger.partition_report()
        for epoch in range(self.start_epoch, self.total_epoch + 1):
            self.train_step(epoch)
        self.logger.partition_report()
        self.logger.time_report("Finish:")
        self.logger.partition_report()

    def save(self, epoch: int):
        self.logger.partition_report()
        self.logger.time_report("Saving:")
        self.checkpoint.save(self.model,
                             self.optimizer,
                             epoch,
                             self.step)
        self.logger.time_report("Saving complete!")
        self.logger.partition_report()

    def train_step(self, epoch: int):
        train_loss: Averager = Averager()
        for batch, (image, target) in enumerate(self.train_loader):
            self.model.train()
            bs = image.size(0)
            image = image.to(self.device)
            target = target.to(self.device)
            pred: Tensor = self.model(image, target[:, :-1])
            loss: Tensor = self.criterion(pred, target)
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            train_loss.update(loss.item(), bs)
            if self.step % self.save_interval == 0:
                self.logger.partition_report()
                self.logger.time_report("epoch {} - step {}:".format(epoch, self.step))
                valid_loss = self.valid_step()
                test_result = self.test_step()
                self.logger.train_report({
                    "train_loss": train_loss.calc(),
                    "valid_loss": valid_loss,
                    "test_result": test_result
                })
                self.logger.partition_report()
                train_loss.clear()
                if self.step > 0:
                    self.save(epoch)
            self.step += 1

    def valid_step(self):
        self.model.eval()
        valid_loss: Averager = Averager()
        with torch.no_grad():
            for batch, (image, target) in enumerate(self.valid_loader):
                bs = image.size(0)
                image = image.to(self.device)
                target = target.to(self.device)
                pred: Tensor = self.model(image, target[:, :-1])
                loss: Tensor = self.criterion(pred, target)
                valid_loss.update(loss.item(), bs)
        return valid_loss.calc()

    def test_step(self):
        self.model.eval()
        test_acc: Averager = Averager()
        test_norm: Averager = Averager()
        with torch.no_grad():
            for batch, (image, target) in enumerate(self.valid_loader):
                bs = image.size(0)
                image = image.to(self.device)
                target = target.to(self.device)
                pred: Tensor = self.model(image, target[:, :-1])
                acc, norm = self._acc(pred, target)
                test_acc.update(acc, bs)
                test_norm.update(norm, bs)
        return {
            "acc": test_acc.calc(),
            "norm": test_norm.calc()
        }

    def _acc(self, pred: Tensor, target: Tensor) -> Tuple:
        n_correct: int = 0
        norm_edit: int = 0
        bs = pred.size(0)
        rp: np.ndarray = pred.softmax(dim=2).cpu().detach().numpy().argmax(axis=2)
        rt: np.ndarray = target.cpu().detach().numpy()
        for i in range(bs):
            p_str: str = self.alphabet.decode(rp[i])
            t_str: str = self.alphabet.decode(rt[i])
            max_len = max(len(p_str), len(t_str))
            if max_len == 0:
                continue
            norm_edit += Levenshtein.distance(p_str, t_str) / max_len
            if p_str == t_str:
                n_correct += 1
        return n_correct, norm_edit

    def load(self):
        state_dict: Dict = self.checkpoint.load()
        if state_dict is not None:
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.start_epoch = state_dict['epoch'] + 1
            self.step = state_dict['step'] + 1
