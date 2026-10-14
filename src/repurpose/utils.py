from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


def load_config(path="configs/smoke.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def paths(config):
    base = ROOT / "data" / "processed" / config["profile"]
    models = ROOT / "models" / config["profile"]
    reports = ROOT / "reports" / config["profile"]
    for p in [base, models, reports]:
        p.mkdir(parents=True, exist_ok=True)
    return base, models, reports


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        if isinstance(x, (Path,)):
            return str(x)
        raise TypeError(type(x).__name__)

    # Missing metrics are represented as null, never nonstandard JSON NaN.
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        return x

    path.write_text(
        json.dumps(clean(obj), ensure_ascii=False, indent=2, default=convert, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
