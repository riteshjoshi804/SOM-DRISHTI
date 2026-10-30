"""On-disk layout of registration runs: ``results/jobs/<job_id>/``.

Each job directory holds ``result.json`` plus the image artifacts. Scripts and
the API share this layout, so a run started from the command line shows up in
the web interface as well. Nothing is ever written next to the input data.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from app import config
from app.registration.schemas import RegistrationResult

RESULT_FILE = "result.json"
_JOB_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def new_job_id(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", label) or "job"
    return f"{safe}__{datetime.now():%Y%m%d-%H%M%S-%f}"[:-3]


def is_valid_job_id(job_id: str) -> bool:
    """Rejects anything that could escape the jobs directory."""
    return bool(_JOB_ID.match(job_id)) and ".." not in job_id


def job_dir(job_id: str) -> Path:
    if not is_valid_job_id(job_id):
        raise ValueError(f"Invalid job id: {job_id!r}")
    return config.JOBS_ROOT / job_id


def save_result(directory: Path, result: RegistrationResult) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RESULT_FILE
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_result(job_id: str) -> dict | None:
    path = job_dir(job_id) / RESULT_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_job_ids() -> list[str]:
    """Finished jobs on disk, newest first."""
    root = config.JOBS_ROOT
    if not root.is_dir():
        return []
    jobs = [p for p in root.iterdir() if p.is_dir() and (p / RESULT_FILE).is_file()]
    return [p.name for p in sorted(jobs, key=lambda p: (p / RESULT_FILE).stat().st_mtime, reverse=True)]
