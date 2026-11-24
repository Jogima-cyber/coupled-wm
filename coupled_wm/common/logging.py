# SPDX-License-Identifier: Apache-2.0
"""Console logging setup and a rank-0 experiment logger (TensorBoard + optional wandb).

wandb is never required: it is imported only when the ``wandb`` config block has
``enabled: true``. When enabled, TensorBoard events are mirrored to wandb
(``sync_tensorboard``), so every scalar goes through the TensorBoard writer. wandb writes its
files next to the TensorBoard dir (``<run_dir>/wandb`` for ``log_dir=<run_dir>/tb``), never
inside it: a copy inside the watched dir would be synced again, recursively.

Config block (all keys optional)::

    wandb:
      enabled: false
      mode: offline        # online | offline | disabled
      project: coupled-wm
      entity: null
      name: null           # defaults to the logger's run_name, else the run directory name
      group: null
      tags: []
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping

_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"


def setup_logging(rank: int = 0, level: int | str = "INFO") -> None:
    """Configure the root logger: ``level`` on rank 0, warnings and errors only elsewhere."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    prefix = f"[rank{rank}]" if rank else ""
    handler.setFormatter(logging.Formatter(prefix + _FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level if rank == 0 else logging.WARNING)

    from coupled_wm.common.distributed import report_launch

    report_launch()  # init_distributed defers its launch summary while logging is unconfigured


class ExperimentLogger:
    """Rank-0 scalar logger. Every method is a no-op on other ranks.

    Args:
        log_dir: TensorBoard event directory (created on rank 0), normally ``<run_dir>/tb``.
        rank: global rank of this process.
        config: run config; written as TensorBoard text and passed to ``wandb.init``.
        wandb_cfg: the ``wandb`` config block (see module docstring).
        run_name: wandb run name when the block sets none; default: the run directory name
            (the parent of a ``tb`` log dir).
        purge_step: the first step this launch logs: 0 for a fresh run, the resumed step
            after a restart. TensorBoard then drops the events at or after it that the
            previous launch wrote after its last checkpoint. ``None`` purges nothing.
    """

    def __init__(
        self,
        log_dir: str | os.PathLike,
        rank: int = 0,
        config: Mapping[str, Any] | None = None,
        wandb_cfg: Mapping[str, Any] | None = None,
        run_name: str | None = None,
        purge_step: int | None = None,
    ):
        self.rank = rank
        self.log_dir = Path(log_dir)
        self.run_name = run_name or (self.log_dir.parent if self.log_dir.name == "tb" else self.log_dir).name
        self.writer = None
        self.wandb_run = None
        if rank != 0:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        wandb_cfg = dict(wandb_cfg or {})
        if wandb_cfg.get("enabled", False):
            self.wandb_run = self._init_wandb(wandb_cfg, config)
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(self.log_dir), purge_step=purge_step)
        if config is not None:
            self.log_config(config)

    def _init_wandb(self, wandb_cfg: dict, config: Mapping[str, Any] | None):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb.enabled is true but wandb is not installed; `pip install wandb` "
                "or set wandb.enabled: false"
            ) from e
        return wandb.init(
            project=wandb_cfg.get("project", "coupled-wm"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name") or self.run_name,
            group=wandb_cfg.get("group"),
            tags=wandb_cfg.get("tags"),
            mode=wandb_cfg.get("mode", "offline"),
            dir=str(self.log_dir.parent),
            config=dict(config) if config is not None else None,
            sync_tensorboard=True,
            resume="allow",
            id=wandb_cfg.get("id"),
        )

    @property
    def active(self) -> bool:
        return self.writer is not None

    def log_config(self, config: Mapping[str, Any]) -> None:
        if not self.active:
            return
        import yaml

        text = yaml.safe_dump(_jsonable(config), sort_keys=False)
        self.writer.add_text("config", "```\n" + text + "\n```", 0)

    def scalar(self, tag: str, value: Any, step: int) -> None:
        if self.active:
            self.writer.add_scalar(tag, float(value), step)

    def scalars(self, values: Mapping[str, Any], step: int, prefix: str = "") -> None:
        if not self.active:
            return
        for tag, value in values.items():
            self.writer.add_scalar(f"{prefix}{tag}", float(value), step)

    def flush(self) -> None:
        if self.active:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)
