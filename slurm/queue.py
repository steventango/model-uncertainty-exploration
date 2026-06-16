"""SLURM queue helpers."""

import os
import subprocess


def _parse_squeue_array_line(line: str) -> int | None:
    line = line.strip()
    if not line or line == "N/A":
        return None
    for sep in ("_", "."):
        if sep in line:
            suffix = line.rsplit(sep, 1)[-1]
            if suffix.isdigit():
                return int(suffix)
    return None


def active_task_ids(job_name: str, *, user: str | None = None) -> set[int]:
    cmd = [
        "squeue",
        "-u",
        user or os.environ.get("USER", ""),
        "-n",
        job_name,
        "-h",
        "-o",
        "%i",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()

    active: set[int] = set()
    for line in result.stdout.splitlines():
        task_id = _parse_squeue_array_line(line)
        if task_id is not None:
            active.add(task_id)
    return active


def array_spec(task_ids: list[int]) -> str:
    return ",".join(str(t) for t in task_ids)
