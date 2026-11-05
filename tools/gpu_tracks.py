"""Initialize independent GPU histories from the immutable pre-track H100 history."""

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_config import get_profile, profile_for_job, read_config
from tools import record_run, rental


def initialize():
    legacy = ROOT / "results/decisions.json"
    records = json.loads(legacy.read_text()) if legacy.exists() else []
    for identifier in read_config()["gpus"]:
        profile = get_profile(identifier)
        selected = [row for row in records
                    if profile_for_job(rental.load(row["id"])).identifier == identifier]
        results = profile.results_directory(ROOT)
        results.mkdir(parents=True, exist_ok=True)
        ledger = results / "decisions.json"
        existing = json.loads(ledger.read_text()) if ledger.exists() else []
        if existing[:len(selected)] != selected and existing:
            raise ValueError("The GPU history differs from the original recorded history.")
        for row in selected:
            source = ROOT / "results/runs" / row["id"]
            target = results / "runs" / row["id"]
            if source.exists() and not target.exists():
                shutil.copytree(source, target)
            record_run.verify_artifacts(row, target)
        if not ledger.exists() or (not existing and selected):
            ledger.write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n")
            existing = selected
        record_run.write_outputs(existing, profile.directory(ROOT) / "results.tsv")
        record_run.write_best(existing, profile)
        print(json.dumps(dict(gpu_id=identifier, records=len(existing),
                              best=next((row["id"] for row in reversed(existing)
                                         if row["status"] == "keep"), None))))


if __name__ == "__main__":
    initialize()
