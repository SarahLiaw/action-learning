"""Seeding and run records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float)+'\n')


def read_json(path):
    return json.loads(Path(path).read_text())
