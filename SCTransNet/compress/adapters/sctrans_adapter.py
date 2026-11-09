"""
SCTransNet Adapter — 4 CNN stage widths (base_channel=32 教师):

  stage0 -> 32   inc / up_decoder1&2 out / gt_conv2
  stage1 -> 64   down_encoder1 / up_decoder3 out / gt_conv3
  stage2 -> 128  down_encoder2 / up_decoder4 out / gt_conv4
  stage3 -> 256  down_encoder3&4 / mtc scale4 / gt_conv5

Transformer 拓扑不变；KV_size = sum(channels)。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch.nn as nn

from compress.groups import PrunableGroup
from model.SCTransNet import Res_block

TEACHER_CHANNELS = (32, 64, 128, 256)


class SCTransAdapter:
    def __init__(self, channels: Sequence[int] = TEACHER_CHANNELS):
        self.channels = list(channels)
        self.nb_filter = self.channels
        self.num_stages = len(self.channels)
        assert self.num_stages == 4

    def stage_of_out_channels(self, out_channels: int) -> int:
        try:
            return self.channels.index(out_channels)
        except ValueError:
            return -1

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

    def get_hook_modules(self, model: nn.Module) -> Dict[int, List[nn.Module]]:
        hooks = {s: [] for s in range(self.num_stages)}
        for _, module in model.named_modules():
            if isinstance(module, Res_block):
                stage = self.stage_of_out_channels(module.conv2.out_channels)
                if stage >= 0:
                    hooks[stage].append(module)
        return hooks
