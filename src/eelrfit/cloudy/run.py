from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from astropy.config.paths import get_cache_dir_path

from .grid import CloudyGrid, read_cloudy_grid
from .spec import GRID_BASENAME, LINE_LIST_NAME, GridSpec

__all__ = [
    "CloudyRunError",
    "available_cpus",
    "count_finished_models",
    "default_cache_dir",
    "grid_key",
    "resolve_executable",
    "run_grid",
]

INPUT_NAME = "input"
RECORD_NAME = "run.json"
SED_NAME = "sed.sed"
GRID_FILE = "grid.nc"


class CloudyRunError(RuntimeError):
    """Running Cloudy failed; the message points to the log."""


def available_cpus() -> int:
    """CPUs this process may use.

    In order: ``$EELRFIT_CPUS``; the CPU affinity mask (Linux; respects batch
    schedulers' cpusets); ``$SLURM_CPUS_PER_TASK``, ``$NSLOTS`` (SGE/UGE) or
    ``$PBS_NUM_PPN``; finally the machine's CPU count.
    """
    if os.environ.get("EELRFIT_CPUS"):
        return max(1, int(os.environ["EELRFIT_CPUS"]))
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    for var in ("SLURM_CPUS_PER_TASK", "NSLOTS", "PBS_NUM_PPN"):
        if os.environ.get(var, "").isdigit():
            return max(1, int(os.environ[var]))
    return os.cpu_count() or 1


def count_finished_models(run_dir: str | os.PathLike) -> int:
    """Models of a running grid that have finished.

    While a grid runs, Cloudy writes each model's output to
    ``grid<index>_<input>.out``, ending with a ``[Stop in ...]`` line when the
    model is done; the files are merged into ``<input>.out`` at the end.
    """
    done = 0
    for name in glob.glob(os.path.join(os.fspath(run_dir), f"grid*_{INPUT_NAME}.out")):
        try:
            with open(name, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                fh.seek(max(0, fh.tell() - 2048))
                if b"[Stop in" in fh.read():
                    done += 1
        except OSError:
            continue
    return done


def default_cache_dir() -> Path:
    """``$EELRFIT_GRID_CACHE`` if set, otherwise ``cloudy-grids`` in EELRFit's cache directory."""
    if os.environ.get("EELRFIT_GRID_CACHE"):
        return Path(os.environ["EELRFIT_GRID_CACHE"]).expanduser()
    return Path(get_cache_dir_path("eelrfit")) / "cloudy-grids"


def resolve_executable(executable: str | os.PathLike | None = None) -> Path:
    """The Cloudy executable: ``executable`` if given, else ``$CLOUDY_EXE``."""
    executable = executable or os.environ.get("CLOUDY_EXE")
    if not executable:
        raise CloudyRunError(
            "No Cloudy executable: pass one or set $CLOUDY_EXE (scripts/install_cloudy.sh builds Cloudy)"
        )
    path = Path(executable).expanduser()
    if not path.is_file():
        raise CloudyRunError(f"Cloudy executable not found: {path}")
    return path.absolute()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grid_key(spec: GridSpec, sed_sha256: str, executable_sha256: str) -> str:
    """Hash identifying a grid's physics and the Cloudy build that computes it.

    The SED enters by its contents (it is copied to a fixed name in the run
    directory), so renaming the file does not change the key.
    """
    payload = {
        "input": spec.render_input(sed_name=SED_NAME),
        "line_list": spec.render_line_list(),
        "sed_sha256": sed_sha256,
        "cloudy_executable_sha256": executable_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_grid(
    spec: GridSpec,
    *,
    executable: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
    force: bool = False,
    timeout: float | None = None,
    ncpus: int | None = None,
    progress_interval: float | None = 60.0,
    report: Callable[[str], None] | None = None,
) -> CloudyGrid:
    """Run a Cloudy grid (or load it from the cache) and return it.

    Each grid runs in its own directory, named by a hash of everything that
    determines the result: the rendered input deck, the line list, the SED
    contents, and the Cloudy executable (by checksum, so a rebuilt or different
    version gives a different key). The directory keeps the input, a copy of the
    SED, Cloudy's output and log, the parsed grid (``grid.nc``), and a
    ``run.json`` record; an identical grid is returned from the cache. Cloudy
    runs the models in parallel itself (by forking), with ``ncpus`` processes.

    Parameters
    ----------
    spec : GridSpec
        Physics of the grid.
    executable : path-like, optional
        Cloudy executable; default ``$CLOUDY_EXE``.
    cache_dir : path-like, optional
        Where run directories live (default `default_cache_dir`).
    force : bool
        Re-run even if a completed run with the same key exists.
    timeout : float, optional
        Seconds before Cloudy is killed.
    ncpus : int, optional
        Processes Cloudy may fork (default `available_cpus`). Not part of the
        cache key, since it does not change the results.
    progress_interval : float, optional
        Seconds between progress reports while running (None to disable).
    report : callable, optional
        Receives progress messages.
    """
    say = report or (lambda message: None)
    exe = resolve_executable(executable)
    exe_sha = _sha256(exe)
    sed_path = spec.sed_path(exe)
    sed_sha = _sha256(sed_path)
    key = grid_key(spec, sed_sha, exe_sha)
    run_dir = Path(cache_dir or default_cache_dir()).expanduser() / key[:16]
    record_path = run_dir / RECORD_NAME

    if run_dir.exists() and not force:
        record = _read_json(record_path)
        if record and record.get("status") == "ok" and (run_dir / GRID_FILE).is_file():
            say(f"Using cached grid {run_dir}")
            return CloudyGrid.read(run_dir / GRID_FILE)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    ncpus = max(1, min(int(ncpus or available_cpus()), spec.n_models))
    deck = spec.render_input(sed_name=SED_NAME, ncpus=ncpus)
    shutil.copyfile(sed_path, run_dir / SED_NAME)
    (run_dir / f"{INPUT_NAME}.in").write_text(deck)
    (run_dir / LINE_LIST_NAME).write_text(spec.render_line_list())
    record: dict[str, Any] = {
        "status": "running",
        "key": key,
        "spec": spec.to_dict(),
        "sed": {"path": str(sed_path), "sha256": sed_sha},
        "n_models": spec.n_models,
        "cloudy_executable": str(exe),
        "cloudy_executable_sha256": exe_sha,
        "ncpus": ncpus,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    _write_json(record_path, record)

    say(f"Running {spec.n_models} Cloudy models on {ncpus} CPU(s) in {run_dir}")
    started = time.monotonic()
    log_path = run_dir / "cloudy.log"
    with (
        open(log_path, "w") as log,
        subprocess.Popen(
            [str(exe), "-r", INPUT_NAME],
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as process,
    ):
        last_report = started
        while True:
            try:
                process.wait(timeout=5.0)
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if timeout is not None and now - started > timeout:
                    process.kill()
                    process.wait()
                    _write_json(
                        record_path,
                        {
                            **record,
                            "status": "failed",
                            "error": f"timed out after {timeout} s",
                        },
                    )
                    raise CloudyRunError(
                        f"Cloudy did not finish within {timeout} s; see {run_dir}"
                    ) from None
                if (
                    progress_interval is not None
                    and now - last_report >= progress_interval
                ):
                    last_report = now
                    done = count_finished_models(run_dir)
                    elapsed = now - started
                    # Models finish in waves of ncpus; the throughput estimate is meaningless before one full wave.
                    eta = (
                        f", about {elapsed / done * (spec.n_models - done) / 60:.0f} min left"
                        if done >= ncpus
                        else ""
                    )
                    say(
                        f"  {done}/{spec.n_models} models finished after {elapsed / 60:.1f} min{eta}"
                    )
    completed = process
    seconds = time.monotonic() - started

    out_path = run_dir / f"{INPUT_NAME}.out"
    grd, lin = run_dir / f"{GRID_BASENAME}.grd", run_dir / f"{GRID_BASENAME}.lin"
    if completed.returncode != 0 or not grd.is_file() or not lin.is_file():
        _write_json(
            record_path,
            {**record, "status": "failed", "returncode": completed.returncode},
        )
        raise CloudyRunError(
            f"Cloudy exited with code {completed.returncode} and the grid output is "
            f"{'present' if grd.is_file() and lin.is_file() else 'missing'}; see {out_path} and {log_path}"
        )

    cloudy_version = _cloudy_version(out_path)
    provenance = {
        **{
            k: record[k]
            for k in (
                "key",
                "spec",
                "sed",
                "cloudy_executable",
                "cloudy_executable_sha256",
                "ncpus",
                "host",
                "started_at",
            )
        },
        "run_dir": str(run_dir),
        "seconds": round(seconds, 1),
    }
    grid = read_cloudy_grid(
        grd,
        lin,
        parameters=[a.name for a in spec.varied],
        fixed=spec.fixed,
        input_deck=deck,
        cloudy_version=cloudy_version,
        provenance=provenance,
    )
    if dict(grid.shape) != spec.shape:
        raise CloudyRunError(
            f"Cloudy produced a grid of shape {grid.shape}, expected {spec.shape}; see {out_path}"
        )
    grid.to_netcdf(run_dir / GRID_FILE)
    n_failed = int(grid.dataset["failed"].sum())
    _write_json(
        record_path,
        {
            **record,
            "status": "ok",
            "seconds": round(seconds, 1),
            "cloudy_version": cloudy_version,
            "n_failed": n_failed,
        },
    )
    say(
        f"Finished in {seconds / 60:.1f} min"
        + (f"; {n_failed} model(s) failed" if n_failed else "")
    )
    return grid


def _cloudy_version(out_path: Path) -> str:
    try:
        with open(out_path, errors="replace") as fh:
            for _, line in zip(range(200), fh):
                match = re.search(r"Cloudy (\d{2}\.\d{2})", line)
                if match:
                    return match.group(1)
    except OSError:
        pass
    return ""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
