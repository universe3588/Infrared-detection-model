from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PrunableGroup:
    group_id: str
    kind: str  # conv_out
    stage_id: int
    channels: int
    module_names: List[str] = field(default_factory=list)
    depends_on: Optional[List[str]] = None
