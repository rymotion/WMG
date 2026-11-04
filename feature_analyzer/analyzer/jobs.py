"""Background jobs (one worker thread: torch + afconvert don't benefit from more) with status."""

from __future__ import annotations

import queue
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    label: str
    state: str = "queued"            # queued | running | done | failed
    message: str = ""
    result: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""
    log: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="analyzer-jobs", daemon=True)
            self._thread.start()

    def submit(self, label: str, fn: Callable[[Callable[[str], None]], dict]) -> Job:
        """`fn(log)` runs in the worker; it returns a dict stored as the job's result."""
        job = Job(id=uuid.uuid4().hex[:10], label=label)
        with self._lock:
            self.jobs[job.id] = job
        self._q.put((job, fn))
        self._ensure_worker()
        return job

    def _worker(self) -> None:
        while True:
            job, fn = self._q.get()
            job.state, job.started_at = "running", _now()

            def log(msg: str, _job=job) -> None:
                _job.message = msg
                _job.log.append(msg)

            try:
                job.result = fn(log) or {}
                job.state = "done"
            except Exception as e:  # noqa: BLE001
                job.state = "failed"
                job.message = f"{type(e).__name__}: {e}"
                job.log.append(traceback.format_exc())
            job.finished_at = _now()
            self._q.task_done()

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def recent(self, n: int = 50) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)[:n]

    def active(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.state in ("queued", "running")]

    def clear_finished(self) -> None:
        with self._lock:
            self.jobs = {k: j for k, j in self.jobs.items() if j.state in ("queued", "running")}


manager = JobManager()
