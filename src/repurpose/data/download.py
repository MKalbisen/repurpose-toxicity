"""Versioned public data downloads with atomic writes and source hashes."""

import hashlib
import time
from pathlib import Path

import requests

from repurpose.utils import ROOT, write_json

DATAVERSE = "https://dataverse.harvard.edu/api/datasets/:persistentId/"
DOI = "doi:10.7910/DVN/IXA7BM"
TOX21 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


def fetch(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            with partial.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def download():
    from repurpose.utils import sha256

    kg = ROOT / "data/raw/primekg/primekg_raw.csv"
    tox = ROOT / "data/raw/tox21/tox21.csv.gz"
    origin = "official_dataverse"
    checksum = None
    version_string = None
    license_meta = "See PrimeKG original source licenses; mirror does not grant new rights."
    official_error = None
    try:
        r = requests.get(DATAVERSE, params={"persistentId": DOI}, timeout=60)
        r.raise_for_status()
        meta = r.json()
        write_json(ROOT / "reports/primekg_dataverse_metadata.json", meta)
        version = meta["data"]["latestVersion"]
        record = next(f["dataFile"] for f in version["files"] if f["dataFile"]["filename"] == "kg.csv")
        kg_url = f"https://dataverse.harvard.edu/api/access/datafile/{record['id']}"
        checksum = record.get("checksum")
        version_string = f"{version['versionNumber']}.{version['versionMinorNumber']}"
        license_meta = version.get("license", version.get("termsOfUse", license_meta))
        fetch(kg_url, kg)
    except requests.RequestException as error:
        # Pinned mirror is a transparent availability fallback, never reported as the
        # latest official release. Its content hash is verified against HF LFS metadata.
        official_error = str(error)
        origin = "pinned_community_mirror"
        commit = "03de2c6613831c66203fb8af8ac7f3fdae6b1887"
        kg_url = f"https://huggingface.co/datasets/hieupth/primekg/resolve/{commit}/org/kg.csv"
        checksum = {
            "type": "SHA-256",
            "value": "0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69",
        }
        version_string = f"mirror commit {commit}; official version unverified"
        print("Dataverse erişilemiyor; hash-sabitli topluluk aynası kullanılıyor.", flush=True)
        fetch(kg_url, kg)
    if checksum:
        h = hashlib.new(checksum["type"].lower().replace("-", ""))
        with kg.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest().lower() != checksum["value"].lower():
            raise ValueError("PrimeKG kaynak checksum doğrulaması başarısız")
    fetch(TOX21, tox)
    write_json(
        ROOT / "reports/data_manifest.json",
        {
            "retrieved_unix": time.time(),
            "primekg_doi": DOI,
            "origin": origin,
            "official_access_error": official_error,
            "primekg_version": version_string,
            "license_metadata": license_meta,
            "sources": [
                {"url": kg_url, "file": str(kg.relative_to(ROOT)), "sha256": sha256(kg)},
                {"url": TOX21, "file": str(tox.relative_to(ROOT)), "sha256": sha256(tox)},
            ],
            "license_note": "PrimeKG software MIT != underlying source licenses; DrugBank is restricted.",
        },
    )
