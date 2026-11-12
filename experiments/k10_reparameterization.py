"""Research-only training branches that fuse to K10's ordinary convolution graph.

Inspired by RepVGG (https://github.com/DingXiaoH/RepVGG), not a port of its recipe.
Each original Conv3x3/BN gains one parallel Conv1x1/BN before the existing ELU.
The new BN scale starts at zero: source weights, buffers and initial function are
preserved. This is a different optimization parameterization, not more deployed
capacity. It has no measured accuracy advantage and is not used by release code.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import fuse_conv_bn_eval

from drone.model import KeyCNN


class FusedKeyCNN(KeyCNN):
    """Experimental v1 deploy factory, distinct from production checkpoints.

    State dictionaries require this factory: the production KeyCNN loader expects
    BN buffers that do not exist here. This is not a released artifact schema.
    """

    def __init__(self) -> None:
        with torch.device("cpu"):
            super().__init__()
        for i, module in enumerate(self.net):
            if isinstance(module, nn.BatchNorm2d):
                self.net[i] = nn.Identity()


class TrainingBranches(nn.Module):
    """Two linear branches with separate running statistics, then one activation."""

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        self.main_conv = copy.deepcopy(conv)
        self.main_bn = copy.deepcopy(bn)
        with torch.device("cpu"):
            self.side_conv = nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=False)
            self.side_bn = nn.BatchNorm2d(conv.out_channels, eps=bn.eps, momentum=bn.momentum)
        self.side_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)
        self.side_bn.to(device=conv.weight.device, dtype=conv.weight.dtype)
        nn.init.zeros_(self.side_bn.weight)
        self.train(conv.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main_bn(self.main_conv(x)) + self.side_bn(self.side_conv(x))

    def fused(self) -> nn.Conv2d:
        """Fold frozen population statistics; equivalence is subject to rounding."""
        if any(module.training for module in self.modules()):
            raise ValueError("Fusion requires every branch module in evaluation mode")
        main = fuse_conv_bn_eval(self.main_conv, self.main_bn)
        side = fuse_conv_bn_eval(self.side_conv, self.side_bn)
        assert main.bias is not None and side.bias is not None
        with torch.no_grad():
            main.weight.add_(F.pad(side.weight, (1, 1, 1, 1)))
            main.bias.add_(side.bias)
        return main


class ReparameterizedKeyCNN(nn.Module):
    """Copy a canonical width-one K10, leaving its owner and RNG untouched."""

    def __init__(self, source: KeyCNN) -> None:
        super().__init__()
        if any(t.device.type != "cpu" for t in source.state_dict().values()):
            raise ValueError("Construct from a CPU source before moving the candidate to a device")
        if any(module.training != source.training for module in source.modules()):
            raise ValueError("Expected a uniform source train/eval mode")
        self.model = copy.deepcopy(source)
        indices = [i for i, module in enumerate(self.model.net) if isinstance(module, nn.Conv2d)]
        shapes = [(1, 16), (16, 16), (16, 32), (32, 32), (32, 64), (64, 64)]
        if len(indices) != len(shapes):
            raise ValueError("Expected six canonical width-one K10 convolutions")
        # One advancing private stream avoids shared random prefixes across side
        # kernels while preserving the caller's initial dropout RNG state.
        with torch.random.fork_rng(devices=[]):
            self._replace_pairs(indices, shapes)
        self.train(source.training)

    def _replace_pairs(self, indices: list[int], shapes: list[tuple[int, int]]) -> None:
        for i, (cin, cout) in zip(indices, shapes, strict=True):
            conv, bn = self.model.net[i], self.model.net[i + 1]
            if (
                not isinstance(conv, nn.Conv2d)
                or not isinstance(bn, nn.BatchNorm2d)
                or (conv.in_channels, conv.out_channels) != (cin, cout)
                or conv.kernel_size != (3, 3)
                or conv.padding != (1, 1)
                or conv.stride != (1, 1)
                or conv.dilation != (1, 1)
                or conv.groups != 1
                or bn.num_features != cout
                or not bn.affine
                or not bn.track_running_stats
            ):
                raise ValueError("Expected canonical Conv3x3/BN pairs")
            self.model.net[i] = TrainingBranches(conv, bn)
            self.model.net[i + 1] = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def to_deploy(self) -> FusedKeyCNN:
        """Return an independent six-Conv3x3 graph; never mutate training state."""
        if any(module.training for module in self.modules()):
            raise ValueError("Deploy conversion requires the entire model in evaluation mode")
        with torch.random.fork_rng(devices=[]):
            result = FusedKeyCNN()
        result.net = copy.deepcopy(self.model.net)
        result.head = copy.deepcopy(self.model.head)
        for i, module in enumerate(result.net):
            if isinstance(module, TrainingBranches):
                result.net[i] = module.fused()
        return result.eval()
