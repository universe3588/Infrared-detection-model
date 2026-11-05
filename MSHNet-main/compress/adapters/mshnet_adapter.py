"""
MSHNet Adapter — 5 stage widths (same scale layout as DNANet nb_filter):

  stage0 -> 16   encoder_0 / decoder_0 / output_0
  stage1 -> 32   encoder_1 / decoder_1 / output_1
  stage2 -> 64   encoder_2 / decoder_2 / output_2
  stage3 -> 128  encoder_3 / decoder_3 / output_3
  stage4 -> 256  middle_layer
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch.nn as nn

from compress.groups import PrunableGroup
from model.MSHNet import ResNet, DEFAULT_CHANNELS

TEACHER_CHANNELS = tuple(DEFAULT_CHANNELS)

# layer name -> input stage ids for concat (None = RGB)
MSH_IN_LAYOUT = {
    "conv_init": [None],
    "encoder_0": [0],
    "encoder_1": [0],
    "encoder_2": [1],
    "encoder_3": [2],
    "middle_layer": [3],
    "decoder_3": [3, 4],
    "decoder_2": [2, 3],
    "decoder_1": [1, 2],
    "decoder_0": [0, 1],
    "output_0": [0],
    "output_1": [1],
    "output_2": [2],
    "output_3": [3],
    "final": [],  # fixed 4-ch fused masks
}


class MSHNetAdapter:
    def __init__(self, channels: Sequence[int] = TEACHER_CHANNELS):
        self.channels = list(channels)
        self.nb_filter = self.channels  # alias for ImportanceEstimator
        self.num_stages = len(self.channels)
        assert self.num_stages == 5

    def enumerate_groups(self, model: nn.Module) -> List[PrunableGroup]:
        groups = []
        for stage, width in enumerate(self.channels):
            names = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and module.out_channels == width:
                    if module.out_channels == 1:
                        continue
                    names.append(name)
            groups.append(
                PrunableGroup(
                    group_id=f"stage{stage}_width",
                    kind="conv_out",
                    stage_id=stage,
                    channels=width,
                    module_names=names,
                )
            )
        return groups

    def stage_of_out_channels(self, out_channels: int) -> int:
        try:
            return self.channels.index(out_channels)
        except ValueError:
            return -1

    def get_hook_modules(self, model: nn.Module) -> Dict[int, List[nn.Module]]:
        """Hook ResNet blocks whose out_channels match a stage width."""
        hooks = {s: [] for s in range(self.num_stages)}
        for _, module in model.named_modules():
            if isinstance(module, ResNet):
                stage = self.stage_of_out_channels(module.conv2.out_channels)
                if stage >= 0:
                    hooks[stage].append(module)
        return hooks
