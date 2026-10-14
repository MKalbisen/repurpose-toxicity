"""Run with the project's Python 3.11 interpreter."""

import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    modules = [
        "torch",  # Load CUDA DLLs before other native libraries on Windows.
        "rdkit",
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "pykeen",
        "streamlit",
        "pyvis",
        "stmol",
        "py3Dmol",
        "requests",
        "yaml",
        "reportlab",
    ]
    failures, versions = {}, {}
    for module in modules:
        try:
            mod = importlib.import_module(module)
            versions[module] = getattr(mod, "__version__", "import OK")
        except Exception as e:  # noqa: BLE001 - report every import failure
            failures[module] = str(e)
    import torch

    out = {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": versions,
        "failures": failures,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "python_311": sys.version_info[:2] == (3, 11),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/environment_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 1 if failures or not out["python_311"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
