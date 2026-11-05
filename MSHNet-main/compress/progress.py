"""Consistent tqdm progress bars with ETA."""

from __future__ import annotations

import sys
import time
from typing import Optional

from tqdm import tqdm


def _use_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


def make_pbar(
    iterable=None,
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    position: Optional[int] = None,
    ncols: int = 120,
):
    kwargs = dict(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        ncols=ncols,
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
    )
    if position is not None:
        kwargs["position"] = position
    if not _use_tty():
        kwargs["mininterval"] = 2.0
    if iterable is None:
        return tqdm(**kwargs)
    return tqdm(iterable, **kwargs)


class StageTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = time.time()
        print(f"\n======== [{name}] start {time.strftime('%F %T')} ========", flush=True)

    def done(self, extra: str = ""):
        dt = time.time() - self.t0
        h, rem = divmod(int(dt), 3600)
        m, s = divmod(rem, 60)
        msg = f"======== [{self.name}] done in {h:02d}:{m:02d}:{s:02d}"
        if extra:
            msg += f" | {extra}"
        msg += " ========"
        print(msg, flush=True)
        return dt
