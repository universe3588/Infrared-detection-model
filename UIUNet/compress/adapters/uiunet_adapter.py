"""
UIU-Net Adapter
---------------
Map UIU-Net into 4 stage-wise prunable groups on *output* widths:

  stage0 -> 64   (enc stage1, dec stage2d/1d, side1/2)
  stage1 -> 128  (enc stage2, dec stage3d, side3, fuse2)
  stage2 -> 256  (enc stage3, dec stage4d, side4, fuse3)
  stage3 -> 512  (enc stage4/5/6, dec stage5d, side5/6, fuse4/5)

mid_ch is NOT scored independently; it is scaled proportionally when rebuilding.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch.nn as nn

from compress.groups import PrunableGroup
from model.uiunet import REBNCONV, DEFAULT_ENC_OUTS, DEFAULT_DEC_OUTS

TEACHER_STAGE_WIDTHS = (64, 128, 256, 512)

# Which unique stage-id each encoder/decoder out belongs to
ENC_STAGE_IDS = (0, 1, 2, 3, 3, 3)   # e1..e6
DEC_STAGE_IDS = (3, 2, 1, 0, 0)       # d5..d1


class UIUNetAdapter:
    def __init__(self, stage_widths: Sequence[int] = TEACHER_STAGE_WIDTHS):
        self.stage_widths = list(stage_widths)
        self.num_stages = len(self.stage_widths)
        assert self.num_stages == 4

    def stage_of_out_channels(self, out_channels: int) -> int:
        try:
            return self.stage_widths.index(out_channels)
        except ValueError:
            return -1

    def enumerate_groups(self, model: nn.Module) -> List[PrunableGroup]:
        groups = []
        for stage, width in enumerate(self.stage_widths):
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
        """Hook REBNCONV whose out_channels match a stage width (for grad/feat)."""
        hooks = {s: [] for s in range(self.num_stages)}
        for _, module in model.named_modules():
            if isinstance(module, REBNCONV):
                stage = self.stage_of_out_channels(module.conv_s1.out_channels)
                if stage >= 0:
                    hooks[stage].append(module)
        return hooks

    @staticmethod
    def default_teacher_layout():
        return {
            "enc_outs": list(DEFAULT_ENC_OUTS),
            "dec_outs": list(DEFAULT_DEC_OUTS),
        }
