"""Resumable preparation commands; never silently start expensive full training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from repurpose.academic.locking import file_lock
from repurpose.academic.state import AcademicState, atomic_json, now
from repurpose.utils import ROOT


def check_tests(files, output):
    output = Path(output)
    xml = output.with_suffix(".xml")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *files, f"--junitxml={xml}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Akademik kabul testleri başarısız: {output}")
    return xml


def foundation(state):
    state.start("A00", "Plan, protokol taslağı ve hash/atomik durum testleri doğrulanıyor.")
    try:
        test_xml = check_tests(
            ["tests/test_academic_state.py"], ROOT / "reports/academic/foundation_tests.log"
        )
        atomic_json(
            ROOT / "reports/academic/foundation_validation.json",
            {"time": now(), "tests_passed": True, "scope": "stage tracking; not scientific performance"},
        )
        state.finish(
            "A00",
            [
                "ACADEMIC_PLAN.md",
                "configs/academic_protocol.yaml",
                "src/repurpose/academic/state.py",
                "src/repurpose/academic/cli.py",
                "scripts/resume_academic.sh",
                "tests/test_academic_state.py",
                "reports/academic/foundation_validation.json",
                test_xml,
            ],
            "Plan/protokol taslağı, atomik JSON, olay günlüğü, artefakt hash ve kesinti testleri geçti.",
        )
    except Exception as error:
        state.fail("A00", str(error))
        raise


def audit_stage(state):
    from repurpose.academic.audit import audit

    state.start("A01", "MVP veri, split, kod ve kaynak kapsamı hesaplanıyor.")
    try:
        audit()
        base = ROOT / "data/processed/smoke"
        evidence = [
            "src/repurpose/academic/audit.py",
            "reports/academic/audit.json",
            "reports/academic/audit.md",
            "reports/academic/assay_label_audit.csv",
            "reports/data_manifest.json",
        ]
        evidence += [
            base / n
            for n in [
                "entities.csv",
                "triples.tsv",
                "train.tsv",
                "validation.tsv",
                "test.tsv",
                "tox21_clean.csv",
                "tox21_features.npz",
                "drug_chemical_mapping.csv",
            ]
        ]
        # Hash every legacy scientific implementation being audited; edits invalidate A01.
        evidence += [
            ROOT / "src/repurpose" / name
            for name in [
                "kg/train.py",
                "data/kg_data.py",
                "data/chemistry.py",
                "data/tox21.py",
                "toxicity/train.py",
                "toxicity/evaluate.py",
                "toxicity/models.py",
            ]
        ]
        state.finish(
            "A01",
            evidence,
            "Read-only bilimsel denetim tamamlandı; bulgular audit.json/md içinde. Kritik açıklar hâlâ A03–A09 iş listesinde.",
        )
    except Exception as error:
        state.fail("A01", str(error))
        raise


def metrics_stage(state):
    state.start("A02", "Drug-only retrieval ve group/paired bootstrap kabul testleri çalışıyor.")
    try:
        xml = check_tests(["tests/test_academic_metrics.py"], ROOT / "reports/academic/metrics_tests.log")
        # Run a validation-only diagnostic on existing smoke models. This checks the
        # adapter; it is not a new final academic result and never reads test triples.
        from repurpose.academic.diagnostic import validation_diagnostic

        validation_diagnostic()
        atomic_json(
            ROOT / "reports/academic/metrics_validation.json",
            {
                "time": now(),
                "tests_passed": True,
                "scope": "metric correctness + legacy validation diagnostic, not final academic test",
            },
        )
        state.finish(
            "A02",
            [
                "src/repurpose/academic/metrics.py",
                "src/repurpose/academic/diagnostic.py",
                "tests/test_academic_metrics.py",
                "reports/academic/metrics_validation.json",
                "reports/academic/validation_diagnostic.json",
                "reports/academic/validation_drug_ranks.csv",
                xml,
            ],
            "Drug-only filtered MRR, canonical hedef tekilleştirme, analytic random baseline, scaffold/paired bootstrap testleri geçti; legacy validation adapter çalıştı.",
        )
    except Exception as error:
        state.fail("A02", str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description="Akademik hazırlık ve kalıcı ilerleme kayıtları")
    parser.add_argument("command", choices=["init", "status", "audit", "metrics", "resume"])
    parser.add_argument("--force", action="store_true", help="Kayıtlı hazırlık adımlarını yeniden çalıştır")
    args = parser.parse_args()
    state = AcademicState()
    if args.command == "status":
        state.refresh()
        for ident, record in state.status().items():
            print(f"{ident} {record['status']:12s} evidence={record['evidence_valid']} {record['name']}")
        return
    # Single running preparatory process per workspace. OS releases flock on crash.
    with file_lock(state.directory / ".run.lock", blocking=False):
        state.initialize()
        if args.command == "init":
            print("Durum kayıtları hazır. Devam: python -m repurpose.academic.cli resume")
            return
        pipeline = [("A00", foundation)]
        if args.command in ["audit", "metrics", "resume"]:
            pipeline.append(("A01", audit_stage))
        if args.command in ["metrics", "resume"]:
            pipeline.append(("A02", metrics_stage))
        for stage, run in pipeline:
            if state.verified(stage) and not args.force:
                print(f"{stage} doğrulanmış; mevcut kanıt korunuyor.", flush=True)
                continue
            print(f"{stage} başladı.", flush=True)
            run(state)
            print(f"{stage} doğrulandı.", flush=True)
        print(
            "Akademik deney akışı: python -m repurpose.academic.experiment --help. Güncel sonuç: reports/academic/REVIEW.md."
        )


if __name__ == "__main__":
    main()
