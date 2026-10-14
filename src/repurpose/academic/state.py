"""Durable stage records. Locks serialize writers; artifacts are fingerprinted."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from repurpose.academic.locking import file_lock
from repurpose.utils import ROOT, sha256

STAGES = [
    ("A00", "Deney ve devam kayıtları", []),
    ("A01", "Bilimsel veri/kod denetimi", ["A00"]),
    ("A02", "Hedef değerlendirme altyapısı", ["A01"]),
    ("A03", "Akademik veri ve split üretimi", ["A02"]),
    ("A04", "CPU/GPU pilot ve development protokolü", ["A03"]),
    ("A05", "Nihai protokol/test kilidi", ["A04"]),
    ("A06", "Toksisite eğitimleri ve baselines", ["A05"]),
    ("A07", "KGE eğitimleri ve baselines", ["A05"]),
    ("A08", "Final test ve istatistik", ["A06", "A07"]),
    ("A09", "Entegrasyon/ablation/dış doğrulama", ["A08"]),
    ("A10", "Akademik rapor ve yeniden üretme", ["A09"]),
]


def now():
    return datetime.now(UTC).isoformat()


def atomic_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class AcademicState:
    def __init__(self, project_root=ROOT):
        self.root = Path(project_root)
        self.directory = self.root / "reports/academic"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "state.json"
        self.journal = self.directory / "events.jsonl"

    @contextmanager
    def lock(self):
        with file_lock(self.directory / ".state.lock"):
            yield

    def read(self):
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "schema_version": 1,
            "created": now(),
            "updated": now(),
            "preferences": {
                "hardware": "RTX 3050 Ti Laptop GPU 4 GB; uzun eğitim başka makinede yapılabilir",
                "academic_target": "not yet specified",
            },
            "stages": {
                i: {
                    "name": title,
                    "depends_on": deps,
                    "status": "planned",
                    "attempts": [],
                    "evidence": [],
                    "summary": "",
                }
                for i, title, deps in STAGES
            },
        }

    def _event(self, kind, stage, message):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"time": now(), "type": kind, "stage": stage, "message": message}, ensure_ascii=False
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    def _persist(self, state):
        state["updated"] = now()
        atomic_json(self.path, state)
        self._render(state)

    def initialize(self):
        with self.lock():
            state = self.read()
            self._persist(state)
            self._event("initialized", None, "Akademik plan ve devam kayıtları hazır.")
        return state

    def fingerprint(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        relative = path.resolve().relative_to(self.root.resolve()).as_posix()
        if not path.is_file():
            raise FileNotFoundError(path)
        return {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}

    def verified(self, stage):
        return self._verified(self.read(), stage)

    def _verified(self, state, stage):
        record = state["stages"][stage]
        if record["status"] != "validated" or not record["evidence"]:
            return False
        for e in record["evidence"]:
            path = self.root / e["path"]
            if not path.is_file() or sha256(path) != e["sha256"]:
                return False
        return all(self._verified(state, dep) for dep in record["depends_on"])

    def start(self, stage, summary):
        with self.lock():
            state = self.read()
            record = state["stages"][stage]
            missing = [dep for dep in record["depends_on"] if not self._verified(state, dep)]
            if missing:
                raise ValueError(f"Önkoşul doğrulanmadı: {missing}")
            # Attempts are distinct; a previous in-progress attempt is preserved.
            if record["status"] == "in_progress" and record["attempts"]:
                record["attempts"][-1].update(status="interrupted", ended=now())
            attempt = f"{stage}-{len(record['attempts']) + 1:03d}"
            record["attempts"].append({"id": attempt, "started": now(), "status": "in_progress"})
            record.update(status="in_progress", summary=summary)
            self._event("started", stage, summary)
            self._persist(state)
            return attempt

    def finish(self, stage, evidence, summary):
        with self.lock():
            state = self.read()
            record = state["stages"][stage]
            if record["status"] != "in_progress":
                raise ValueError("Aşama önce başlatılmalıdır")
            if not evidence:
                raise ValueError("Kanıtsız aşama doğrulanamaz")
            missing = [dep for dep in record["depends_on"] if not self._verified(state, dep)]
            if missing:
                raise ValueError(f"Önkoşul çalışma sırasında değişti: {missing}")
            fingerprints = [self.fingerprint(p) for p in evidence]
            record.update(status="validated", evidence=fingerprints, summary=summary)
            record["attempts"][-1].update(status="validated", ended=now())
            self._event("validated", stage, summary)
            self._persist(state)

    def fail(self, stage, reason):
        with self.lock():
            state = self.read()
            record = state["stages"][stage]
            record.update(status="failed", summary=reason)
            if record["attempts"]:
                record["attempts"][-1].update(status="failed", ended=now())
            self._event("failed", stage, reason)
            self._persist(state)

    def note(self, message):
        with self.lock():
            state = self.read()
            state.setdefault("notes", []).append({"time": now(), "message": message})
            self._event("note", None, message)
            self._persist(state)

    def status(self):
        state = self.read()
        return {i: {**r, "evidence_valid": self._verified(state, i)} for i, r in state["stages"].items()}

    def refresh(self):
        with self.lock():
            self._persist(self.read())

    def _render(self, state):
        lines = [
            "# Akademik çalışma ilerlemesi",
            "",
            f"Son güncelleme (UTC): {state['updated']}",
            "Ana plan: ACADEMIC_PLAN.md. Bu dosya reports/academic/state.json içinden üretilir.",
            "",
            "## Kullanıcı tercihleri",
            "",
            "- Donanım: " + state["preferences"]["hardware"],
            "- Akademik hedef türü henüz belirtilmedi; geçici kapsam retrospektif tez/makale çalışması.",
            "",
            "## Aşama durumu",
            "",
            "| ID | İş | Durum | Kanıt kontrolü | Sonuç |",
            "|---|---|---|---|---|",
        ]
        next_stage = None
        for i, record in state["stages"].items():
            verified = self._verified(state, i)
            label = "geçerli" if verified else "bekliyor/değişmiş"
            lines.append(
                f"| {i} | {record['name']} | {record['status']} | {label} | {record['summary'].replace(chr(10), ' ').replace('|', '/')} |"
            )
            if next_stage is None and not verified:
                next_stage = i
        lines += [
            "",
            "## Kaldığımız nokta",
            "",
            f"İlk tamamlanmamış veya yeniden doğrulanması gereken aşama: **{next_stage or 'yok'}**.",
            "`source .venv/bin/activate`",
            "`python -m repurpose.academic.cli status`",
            "`python -m repurpose.academic.cli resume`",
            "",
            "Resume yalnız uygulanmış hazırlık/denetim adımlarını yürütür; henüz kodlanmamış uzun eğitimleri başlatmaz.",
            "Güncel çalışma ve komutlar: reports/academic/REVIEW.md ve docs/academic/REMOTE_TRAINING.md.",
            "Akademik veri/pilot/eğitim: python -m repurpose.academic.experiment --help. Smoke sonuçları final sonuç değildir.",
            "",
            "## Değişiklik ve kesinti kaydı",
            "",
            "Her başlangıç/başarısızlık/doğrulama reports/academic/events.jsonl içine UTC zamanı ile eklenir.",
            "State evidence listesinde dosya yolları, byte sayıları ve SHA256 vardır; dosya değişirse eski doğrulama geçersiz olur.",
            "Yeni MLP/KGE eğitimlerinde epoch sonunda optimizer/RNG checkpoint vardır. Klasik model aileleri tamamlanınca saklanır.",
        ]
        added = sorted({e["path"] for record in state["stages"].values() for e in record["evidence"]})
        lines += ["", "## Kayıtlı dosyalar", ""]
        lines += [f"- `{name}`" for name in added]
        lines += ["", "## Son çalışma notları", ""]
        lines += [f"- {n['time']}: {n['message']}" for n in state.get("notes", [])[-12:]]
        (self.root / "ACADEMIC_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
