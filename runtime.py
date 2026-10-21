"""Runtime supervision, resource limits, subprocess isolation, and cleanup."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable


def set_thread_limits(max_threads: int = 8) -> None:
    """Restrict math and ML library thread pools to prevent CPU oversubscription."""
    threads_str = str(max_threads)
    for var in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[var] = threads_str


@dataclass
class SubprocessResult:
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    error: str | None = None


def run_subprocess(
    cmd: list[str],
    timeout: float | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Execute command in subprocess with strict timeout, tree cleanup, and capped I/O."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd) if cwd else None,
            env=run_env,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            duration = time.time() - t0
            return SubprocessResult(
                exit_code=proc.returncode,
                stdout=stdout[-12000:] if stdout else "",
                stderr=stderr[-12000:] if stderr else "",
                duration=duration,
            )
        except subprocess.TimeoutExpired:
            # Kill process and any child processes
            _kill_proc_tree(proc)
            duration = time.time() - t0
            return SubprocessResult(
                exit_code=-1,
                stdout="",
                stderr="",
                duration=duration,
                error=f"Timeout expired after {timeout} seconds",
            )
    except Exception as exc:
        duration = time.time() - t0
        return SubprocessResult(
            exit_code=-1,
            stdout="",
            stderr="",
            duration=duration,
            error=str(exc),
        )


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Kill process and child processes best-effort."""
    try:
        import psutil
        p = psutil.Process(proc.pid)
        for child in p.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        p.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=2.0)
    except Exception:
        pass


class BudgetSupervisor:
    """Tracks global execution budget, timeout boundaries, and retries."""

    def __init__(self, total_budget: float = 10800.0, reserve_margin: float = 30.0):
        self.total_budget = total_budget
        # Ensure reserve_margin doesn't consume all budget when testing with small budgets
        self.reserve_margin = min(reserve_margin, max(0.0, total_budget * 0.05))
        self.start_time = time.time()

    def time_left(self) -> float:
        """Remaining total execution time in seconds."""
        elapsed = time.time() - self.start_time
        return max(0.0, self.total_budget - elapsed)

    def effective_budget(self) -> float:
        """Time left before the emergency reserve margin."""
        return max(0.0, self.time_left() - self.reserve_margin)

    def has_budget(self, needed_seconds: float) -> bool:
        """Check if at least needed_seconds remain within effective budget."""
        return self.effective_budget() >= needed_seconds

    def step(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        timeout: float | None = None,
        retries: int = 0,
        **kwargs: Any,
    ) -> Any | None:
        """Execute a step with bounded timeout and retries, respecting global budget."""
        attempts = 0
        max_attempts = max(1, retries + 1)

        while attempts < max_attempts:
            attempts += 1
            left = self.effective_budget()
            if left <= 0.1:
                return None

            step_timeout = min(timeout, left) if timeout is not None else left
            if step_timeout <= 0.05:
                return None

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fn, *args, **kwargs)
                try:
                    return future.result(timeout=step_timeout)
                except (FuturesTimeoutError, TimeoutError):
                    # Timeout reached
                    continue
                except Exception:
                    # Exception during execution
                    continue

        return None


def bounded_cleanup(work_dir: Path, preserve_names: set[str] | None = None) -> None:
    """Best-effort cleanup of scratch files without deleting crucial artifacts."""
    if preserve_names is None:
        preserve_names = {"submission.csv", "reserve.csv"}

    if not work_dir.exists():
        return

    for item in work_dir.glob("trial_*"):
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.name not in preserve_names:
                item.unlink(missing_ok=True)
        except Exception:
            pass
