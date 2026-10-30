"""Background registration jobs.

Jobs run one at a time on a worker thread (SIFT extraction already uses
several threads, so running jobs in parallel would only thrash the CPU).
Finished results are persisted under ``results/jobs/<job_id>/`` - the same
place the command-line scripts write to - so they survive a server restart.
"""
from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel

from app.config import PipelineConfig
from app.registration import store
from app.registration.schemas import RegistrationResult

JobStateName = Literal["queued", "running", "done", "error"]
ERROR_FILE = "error.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobState(BaseModel):
    job_id: str
    scene_id: str | None
    label: str
    state: JobStateName
    progress: float = 0.0
    message: str = ""
    created_at: str
    finished_at: str | None = None
    status: str | None = None  # SUCCESS / LOW_CONFIDENCE / FAILED once finished
    error: str | None = None


Runner = Callable[[Path, Callable[[str, float], None]], RegistrationResult]


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="registration")

    def submit(self, label: str, scene_id: str | None, run: Runner) -> JobState:
        job_id = store.new_job_id(label)
        state = JobState(job_id=job_id, scene_id=scene_id, label=label, state="queued", created_at=_now())
        with self._lock:
            self._jobs[job_id] = state
        self._executor.submit(self._execute, job_id, run)
        return state.model_copy()

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = current.model_copy(update=changes)

    def _execute(self, job_id: str, run: Runner) -> None:
        out = store.job_dir(job_id)
        self._update(job_id, state="running", message="starting", progress=0.0)

        def progress(message: str, fraction: float) -> None:
            self._update(job_id, message=message, progress=round(float(fraction), 3))

        try:
            result = run(out, progress)
            store.save_result(out, result)
            self._update(
                job_id, state="done", progress=1.0, message="done", status=result.status, finished_at=_now()
            )
        except Exception as exc:  # reported to the client as a FAILED registration
            out.mkdir(parents=True, exist_ok=True)
            detail = {
                "status": "FAILED",
                "failure_reasons": [f"registration exception: {type(exc).__name__}: {exc}"],
                "traceback": traceback.format_exc(),
            }
            (out / ERROR_FILE).write_text(json.dumps(detail, indent=2), encoding="utf-8")
            self._update(
                job_id,
                state="error",
                message="registration exception",
                status="FAILED",
                error=detail["failure_reasons"][0],
                finished_at=_now(),
            )

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id].model_copy()
        return self._from_disk(job_id)

    def _from_disk(self, job_id: str) -> JobState | None:
        if not store.is_valid_job_id(job_id):
            return None
        directory = store.job_dir(job_id)
        result_path = directory / store.RESULT_FILE
        error_path = directory / ERROR_FILE
        if result_path.is_file():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            stamp = datetime.fromtimestamp(result_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            return JobState(
                job_id=job_id,
                scene_id=data.get("scene_id"),
                label=data.get("scene_id") or job_id,
                state="done",
                progress=1.0,
                message="done",
                created_at=stamp,
                finished_at=stamp,
                status=data.get("status"),
            )
        if error_path.is_file():
            data = json.loads(error_path.read_text(encoding="utf-8"))
            stamp = datetime.fromtimestamp(error_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            return JobState(
                job_id=job_id,
                scene_id=None,
                label=job_id,
                state="error",
                progress=1.0,
                message="registration exception",
                created_at=stamp,
                finished_at=stamp,
                status="FAILED",
                error=(data.get("failure_reasons") or ["registration exception"])[0],
            )
        return None

    def list(self) -> list[JobState]:
        with self._lock:
            live = {j.job_id: j.model_copy() for j in self._jobs.values()}
        for job_id in store.list_job_ids():
            if job_id not in live:
                state = self._from_disk(job_id)
                if state is not None:
                    live[job_id] = state
        return sorted(live.values(), key=lambda j: j.created_at, reverse=True)


def merge_config(overrides: dict | None) -> PipelineConfig:
    """Default configuration with a (possibly nested, partial) override applied."""
    base = PipelineConfig().model_dump()

    def merge(into: dict, extra: dict) -> dict:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(into.get(key), dict):
                merge(into[key], value)
            else:
                into[key] = value
        return into

    return PipelineConfig.model_validate(merge(base, overrides or {}))
