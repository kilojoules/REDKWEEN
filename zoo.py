"""Disk-based checkpoint zoo for LLM adapter checkpoints.

Modeled on AI-Plays-Tag/trainer/train_zoo.py:OpponentZoo, but stores
adapter paths on disk rather than in-memory state dicts (LLMs don't fit).

One CheckpointZoo instance manages either victim or adversary checkpoints.
"""

from __future__ import annotations

import os
import random
import re
import shutil
from dataclasses import dataclass, field


@dataclass
class _Entry:
    round_num: int
    adapter_path: str


class CheckpointZoo:
    """Manages a zoo of adapter checkpoints on disk."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._entries: list[_Entry] = []

    def add(self, round_num: int, adapter_path: str) -> None:
        """Add a checkpoint to the zoo.

        The adapter directory is expected to already exist on disk
        (e.g. from checkpoint_adapters()). We just record the path.
        FIFO eviction when exceeding max_size.
        """
        self._entries.append(_Entry(round_num=round_num, adapter_path=adapter_path))

        if len(self._entries) > self.max_size:
            self._entries.pop(0)

    def sample(self) -> str | None:
        """Sample a random adapter path from the zoo (uniform).

        Returns None if the zoo is empty.
        """
        if not self._entries:
            return None
        entry = random.choice(self._entries)
        return entry.adapter_path

    @classmethod
    def from_checkpoints_dir(cls, path: str, role: str = "victim",
                             max_size: int = 50) -> CheckpointZoo:
        """Rebuild a zoo from an existing checkpoints directory.

        Scans *path*/round_*/role/ for adapter directories and adds them
        in round order.

        Args:
            path: Root checkpoints directory (e.g. "experiments/A030_buffered/checkpoints")
            role: Subdirectory name within each round ("victim" or "adversary")
            max_size: Maximum zoo capacity
        """
        zoo = cls(max_size=max_size)

        if not os.path.isdir(path):
            return zoo

        round_dirs = []
        for entry in os.listdir(path):
            match = re.match(r"round_(\d+)$", entry)
            if match:
                round_num = int(match.group(1))
                adapter_dir = os.path.join(path, entry, role)
                adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
                if os.path.exists(adapter_file):
                    round_dirs.append((round_num, adapter_dir))

        # Add in round order
        for round_num, adapter_dir in sorted(round_dirs):
            zoo.add(round_num, adapter_dir)

        return zoo

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        rounds = [e.round_num for e in self._entries]
        return f"CheckpointZoo(size={len(self)}, rounds={rounds})"
