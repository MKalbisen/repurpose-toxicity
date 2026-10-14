"""Content-addressed protocol and explicit final-test access journal."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import yaml

from repurpose.academic.prepare import digest, verify_bundle
from repurpose.academic.state import atomic_json, now
from repurpose.utils import ROOT, sha256


def read_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "configs/academic_run.yaml").read_text(encoding="utf-8"))


def source_hashes():
    files = sorted((ROOT / "src/repurpose").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}


def snapshot(config):
    base = ROOT / "data/processed" / config["dataset"]
    return {
        "packages": {
            name: importlib.metadata.version(name).split("+")[0]
            for name in ["torch", "pykeen", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "rdkit"]
        },
        "config": config,
        "code": source_hashes(),
        "data": {name: digest(verify_bundle(base / name)) for name in ["toxicity", "kg"]},
    }


def freeze(config, run):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "protocol_lock.json").exists():
        return verify_lock(config, run)
    snap = snapshot(config)
    lock = {
        "created": now(),
        "snapshot": snap,
        "sha256": digest(snap),
        "scope": "retrospective benchmark; historical test exposure disclosed; no clinical claims",
    }
    atomic_json(run / "protocol_lock.json", lock)
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (run / "environment.txt").write_text(packages.stdout, encoding="utf-8")
    return lock


def verify_lock(config, run):
    path = Path(run) / "protocol_lock.json"
    if not path.exists():
        raise ValueError("Freeze protocol before training or final evaluation")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if digest(snapshot(config)) != lock["sha256"]:
        raise ValueError("Protocol/code/data changed after lock: create a new run, never reuse final test")
    return lock


def final_access(config, run, domain, artifacts):
    """Record the exact selected artifacts BEFORE opening the held-out test."""
    lock = verify_lock(config, run)
    run = Path(run)
    marker = run / f"{domain}_test_access.json"
    selected = {str(Path(p).relative_to(run).as_posix()): sha256(p) for p in artifacts}
    if marker.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior["selected"] != selected or prior["protocol"] != lock["sha256"]:
            raise ValueError("Selected artifacts changed after test access")
        return prior
    record = {
        "first_access": now(),
        "protocol": lock["sha256"],
        "selected": selected,
        "domain": domain,
        "retry_policy": "same artifacts only; interrupted evaluations may resume",
    }
    atomic_json(marker, record)
    return record
