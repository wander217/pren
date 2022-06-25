from torch import nn, Tensor
from typing import Optional, Callable, List, Tuple
from functools import partial
import math
import torch
import torch.fx


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@torch.fx.wrap
def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError("drop probability has to be between 0 and 1, but got {}".format(p))
    if mode not in ["batch", "row"]:
        raise ValueError("mode has to be either 'batch' or 'row', but got {}".format(mode))
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate).div_(survival_rate)
    return input * noise


class ConvNormActivation(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride: int = 1,
            padding=None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
            dilation: int = 1,
            inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            dilation=dilation,
                            groups=groups,
                            bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            activation: Callable[..., nn.Module] = nn.ReLU,
            scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'p=' + str(self.p)
        tmpstr += ', mode=' + str(self.mode)
        tmpstr += ')'
        return tmpstr


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(self,
                 expand_ratio: float,
                 kernel: int,
                 stride: int,
                 input_channels: int,
                 out_channels: int,
                 num_layers: int,
                 width_mult: float,
                 depth_mult: float,
                 padding: Optional[int] = None) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)
        self.padding = padding

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'expand_ratio={expand_ratio}'
        s += ', kernel={kernel}'
        s += ', stride={stride}'
        s += ', input_channels={input_channels}'
        s += ', out_channels={out_channels}'
        s += ', num_layers={num_layers}'
        s += ')'
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(self,
                 cnf: MBConvConfig,
                 stochastic_depth_prob: float,
                 norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation) -> None:
        super().__init__()

        if isinstance(cnf.stride, int) and not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels,
                                             expanded_channels,
                                             kernel_size=1,
                                             norm_layer=norm_layer,
                                             activation_layer=activation_layer))

        # depthwise
        layers.append(ConvNormActivation(expanded_channels,
                                         expanded_channels,
                                         kernel_size=cnf.kernel,
                                         stride=cnf.stride,
                                         groups=expanded_channels,
                                         norm_layer=norm_layer,
                                         padding=cnf.padding,
                                         activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels,
                               squeeze_channels,
                               activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(ConvNormActivation(expanded_channels,
                                         cnf.out_channels,
                                         kernel_size=1,
                                         norm_layer=norm_layer,
                                         activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class GateConv(nn.Module):
    def __init__(self, n_input: int, n_output: int, d_input: int, d_output: int, dropout: float = 0.1):
        """
            Tổng hợp feature thành chuỗi feature mới theo cả chiều sâu lẫn chiều rộng
            :param n_input: Số lượng feature input
            :param n_output: Số lượng feature output
            :param d_input: Độ dài của feature input
            :param d_output: Độ dài của feature output
        """
        super().__init__()
        self.conv: nn.Module = nn.Conv1d(n_input, n_output, 1)
        self.fc: nn.Module = nn.Sequential(
            nn.Linear(d_input, d_output),
            nn.Dropout(p=dropout),
            nn.Hardswish(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        """
            :param x: (B,n_input,d_input)
            :return: (B, n_output, d_output)
        """
        output: Tensor = self.conv(x)
        output = self.fc(output)
        return output


class PoolAggregation(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, d_output: int):
        """
            Trích thuộc tính toàn cục của feature
        """
        super().__init__()
        self.layer_list: nn.ModuleList = nn.ModuleList([])
        for _ in range(n_output):
            self.layer_list.append(nn.Sequential(
                nn.Conv2d(n_input, n_hidden, (3, 3), (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(n_hidden, momentum=0.01, eps=0.001),
                nn.Hardswish(inplace=True),
                nn.Conv2d(n_hidden, d_output, (3, 3), (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(d_output, momentum=0.01, eps=0.001)
            ))
        self.pool: nn.Module = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        """
            :param x: (B, n_input, H, W)
            :return: (B, n_output , d_output)
        """
        feature: List = []
        bs = x.size(0)
        for layer in self.layer_list:
            f: Tensor = self.pool(layer(x))
            feature.append(f.view(bs, 1, -1))
        output: Tensor = torch.cat(feature, dim=1)
        return output


class WeightAggregation(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, d_output: int):
        """
            Tạo lên một score cho vùng chứa chữ
        """
        super().__init__()
        self.conv_n: nn.Module = nn.Sequential(
            nn.Conv2d(n_input, n_input, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(n_input, momentum=0.01, eps=0.001),
            nn.Hardswish(inplace=True),
            nn.Conv2d(n_input, n_output, (1, 1), bias=False),
            nn.BatchNorm2d(n_output, momentum=0.01, eps=0.001),
            nn.Sigmoid()
        )
        self.conv_d: nn.Module = nn.Sequential(
            nn.Conv2d(n_input, n_hidden, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(n_hidden, momentum=0.01, eps=0.001),
            nn.Hardswish(inplace=True),
            nn.Conv2d(n_hidden, d_output, (1, 1), bias=False),
            nn.BatchNorm2d(d_output, momentum=0.01, eps=0.001),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
            :param x: (B, n_input, H, W)
            :return: (B, n_output , d_output)
        """
        n_map: Tensor = self.conv_n(x)  # (B, n_output, h,w)
        d_map: Tensor = self.conv_d(x)  # (B, d_output, h,w)
        output: Tensor = torch.bmm(n_map.flatten(2),
                                   d_map.flatten(2).permute(0, 2, 1))
        return output
