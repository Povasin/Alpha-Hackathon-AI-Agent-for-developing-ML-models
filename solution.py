"""Entry point and pipeline orchestrator for AutoML solution (Scheme Y-v2)."""

import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

from agent import Journal, QwenAgent, validate_action
from experiments import (
    compile_recipe_code,
    resolve_task_spec,
    split_data,
)
from runtime import BudgetSupervisor, bounded_cleanup, run_subprocess, set_thread_limits
from submission import (
    CandidateStore,
    atomic_publish,
    create_reserve_submission,
    validate_submission,
)


def run_solution(task_path_arg: str = "task.md") -> int:
    # 0. Ensure UTF-8 output streams
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # 1. Start supervisor and set thread limits before heavy workloads
    t0 = time.time()
    supervisor = BudgetSupervisor(total_budget=10800.0, reserve_margin=45.0)
    set_thread_limits(max_threads=8)

    task_path = Path(task_path_arg).resolve()
    base_dir = task_path.parent
    work_dir = base_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    submission_path = base_dir / "submission.csv"

    print(f"[{time.strftime('%X')}] Starting AutoML solution for {task_path}")

    # 2. Resolve task specification and contract
    try:
        task_spec = resolve_task_spec(task_path, base_dir)
        print(
            f"[{time.strftime('%X')}] Task resolved: {task_spec.problem_type}, "
            f"target={task_spec.target_column}, id={task_spec.id_column}, "
            f"metric={task_spec.metric_name} ({task_spec.metric_direction})"
        )
    except Exception as exc:
        print(f"[{time.strftime('%X')}] Failed to resolve task spec: {exc}", file=sys.stderr)
        return 1

    # 3. Create early format reserve and publish atomically before training
    reserve_path = work_dir / "reserve.csv"
    try:
        create_reserve_submission(
            task_spec=task_spec,
            test_path=task_spec.test_path,  # type: ignore
            sample_path=task_spec.sample_path,
            out_path=reserve_path,
        )
        ok, msg = validate_submission(reserve_path, task_spec, task_spec.test_path)  # type: ignore
        if not ok:
            print(f"[{time.strftime('%X')}] Reserve validation failed: {msg}", file=sys.stderr)
            return 1
        atomic_publish(reserve_path, submission_path)
        print(f"[{time.strftime('%X')}] Early format reserve published to {submission_path}")
    except Exception as exc:
        print(f"[{time.strftime('%X')}] Failed to create/publish reserve: {exc}", file=sys.stderr)
        return 1

    # 4. Separate validation and confirmation splits
    try:
        train_df = pd.read_csv(task_spec.train_path)  # type: ignore
        split = split_data(
            train_df,
            target_col=task_spec.target_column,
            problem_type=task_spec.problem_type,
            confirmation_ratio=0.15,
            val_ratio=0.20,
            seed=42,
        )
        print(
            f"[{time.strftime('%X')}] Data split: train={len(split.train_indices)}, "
            f"val={len(split.val_indices)}, confirmation={len(split.confirmation_indices)}"
        )
    except Exception as exc:
        print(f"[{time.strftime('%X')}] Failed to prepare data splits: {exc}", file=sys.stderr)
        return 0  # Reserve is already published, return safe 0

    split_file = work_dir / "split.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_indices": split.train_indices.tolist(),
                "val_indices": split.val_indices.tolist(),
                "confirmation_indices": split.confirmation_indices.tolist(),
            },
            f,
        )

    # 5. Initialize CandidateStore, Journal, and Agent
    candidate_store = CandidateStore(work_dir / "candidates", submission_path)
    journal = Journal(metric_direction=task_spec.metric_direction)
    agent = QwenAgent()
    if not agent.is_available:
        print(f"[{time.strftime('%X')}] [LOCAL DEGRADATION] OPENAI_API_KEY not set. Using LocalAgentController.")
    else:
        print(f"[{time.strftime('%X')}] Using Qwen model {agent.model}")

    # 6. AutoResearch exploration loop
    max_iterations = 8
    for iteration in range(1, max_iterations + 1):
        if not supervisor.has_budget(90.0):
            print(f"[{time.strftime('%X')}] Budget limit reached. Freezing search.")
            break

        action, mode_info = agent.propose_action(journal, task_spec)
        print(f"[{time.strftime('%X')}] Iteration {iteration} ({mode_info}): action={action.get('name')}")

        if action.get("name") == "stop_search":
            print(f"[{time.strftime('%X')}] Agent stopped search: {action.get('arguments', {}).get('reason')}")
            break

        ok, msg = validate_action(action)
        if not ok:
            print(f"[{time.strftime('%X')}] Action validation rejected: {msg}")
            journal.record_node(
                parent_id=journal.get_best_node().id if journal.get_best_node() else 0,
                hypothesis="Invalid action",
                status="error",
                error=msg,
            )
            continue

        recipe = action["arguments"]["recipe"]
        hypothesis = action["arguments"]["hypothesis"]

        trial_dir = work_dir / f"trial_{iteration}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        script_path = trial_dir / "trainer.py"

        # Compile recipe to Python script
        try:
            code = compile_recipe_code(recipe, task_spec)
            script_path.write_text(code, encoding="utf-8")
        except Exception as exc:
            journal.record_node(hypothesis=hypothesis, recipe=recipe, status="error", error=str(exc))
            continue

        # Execute trial via supervisor and subprocess
        step_timeout = min(180.0, supervisor.effective_budget())
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            str(task_spec.train_path),
            str(task_spec.test_path),
            str(split_file),
            str(trial_dir),
        ]

        print(f"[{time.strftime('%X')}] Running trial {iteration} (timeout={step_timeout:.1f}s)...")
        res = run_subprocess(cmd, timeout=step_timeout, cwd=trial_dir)

        eval_result_path = trial_dir / "eval_result.json"
        test_pred_path = trial_dir / "test_predictions.csv"

        if res.exit_code == 0 and eval_result_path.exists() and test_pred_path.exists():
            try:
                with open(eval_result_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                metric_val = float(eval_data.get("metric", 0.0))

                valid, val_msg = validate_submission(test_pred_path, task_spec, task_spec.test_path)  # type: ignore
                if valid:
                    node = journal.record_node(
                        node_id=iteration,
                        parent_id=journal.get_best_node().id if journal.get_best_node() else 0,
                        hypothesis=hypothesis,
                        recipe=recipe,
                        status="success",
                        metric=metric_val,
                        csv_path=str(test_pred_path),
                        duration=res.duration,
                    )
                    # Promote candidate and publish if best
                    candidate_store.save_candidate(iteration, test_pred_path, metric_val, task_spec)
                    if candidate_store.best_candidate and candidate_store.best_candidate.candidate_id == iteration:
                        atomic_publish(test_pred_path, submission_path)
                        print(f"[{time.strftime('%X')}] >>> NEW BEST ({task_spec.metric_name}={metric_val:.4f}) published!")
                    else:
                        print(f"[{time.strftime('%X')}] Trial {iteration} metric: {metric_val:.4f}")
                else:
                    journal.record_node(hypothesis=hypothesis, recipe=recipe, status="error", error=val_msg)
                    print(f"[{time.strftime('%X')}] Trial {iteration} invalid submission: {val_msg}")
            except Exception as exc:
                journal.record_node(hypothesis=hypothesis, recipe=recipe, status="error", error=str(exc))
                print(f"[{time.strftime('%X')}] Trial {iteration} result parsing failed: {exc}")
        else:
            err = res.error or res.stderr[-500:] or f"Exit code {res.exit_code}"
            journal.record_node(hypothesis=hypothesis, recipe=recipe, status="error", error=err)
            print(f"[{time.strftime('%X')}] Trial {iteration} failed: {err}")

    # 7. Finalization and cleanup
    best = candidate_store.best_candidate
    if best:
        print(f"[{time.strftime('%X')}] Finalist candidate {best.candidate_id} with metric {best.metric_value:.4f}")
        candidate_store.publish_best()
    else:
        print(f"[{time.strftime('%X')}] No successful candidate found. Keeping format reserve.")

    bounded_cleanup(work_dir)
    print(f"[{time.strftime('%X')}] Complete! Total duration: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else "task.md"
    sys.exit(run_solution(task_arg))
