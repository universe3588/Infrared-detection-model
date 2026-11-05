from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PrunableGroup:
    group_id: str
    kind: str
    stage_id: int
    channels: int
    module_names: List[str] = field(default_factory=list)
