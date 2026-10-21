"""Unit tests for baseline AutoML solution (Y-v2 scheme)."""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


class TestRuntimeSupervisor(unittest.TestCase):
    def test_budget_decrement_and_timeout(self):
        from runtime import BudgetSupervisor
        sup = BudgetSupervisor(total_budget=1.0)
        self.assertLessEqual(sup.time_left(), 1.0)
        self.assertTrue(sup.has_budget(0.1))
        
        # Test step with timeout
        def slow_fn():
            import time
            time.sleep(0.5)
            return "done"
            
        result = sup.step("slow_task", slow_fn, timeout=0.05, retries=1)
        self.assertIsNone(result)

    def test_bounded_retries(self):
        from runtime import BudgetSupervisor
        sup = BudgetSupervisor(total_budget=5.0)
        attempts = 0

        def failing_fn():
            nonlocal attempts
            attempts += 1
            raise ValueError("Intentional error")

        result = sup.step("failing_task", failing_fn, timeout=1.0, retries=2)
        self.assertIsNone(result)
        self.assertEqual(attempts, 3)  # Initial + 2 retries

    def test_subprocess_timeout_and_error(self):
        from runtime import run_subprocess
        # Test error code
        res = run_subprocess(["python", "-c", "import sys; sys.exit(42)"], timeout=5.0)
        self.assertEqual(res.exit_code, 42)
        # Test timeout
        res_timeout = run_subprocess(
            ["python", "-c", "import time; time.sleep(10)"], timeout=0.1
        )
        self.assertIsNotNone(res_timeout.error)
        self.assertIn("timeout", res_timeout.error.lower())


class TestSubmissionAndReserve(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)
        # Create dummy test.csv
        self.test_path = self.work_dir / "test.csv"
        df = pd.DataFrame({"id": [10, 20, 30], "f1": [1.0, 2.0, 3.0]})
        df.to_csv(self.test_path, index=False)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_validate_submission(self):
        from experiments import TaskSpec
        from submission import validate_submission

        spec = TaskSpec(
            problem_type="regression",
            id_column="id",
            target_column="site_eui",
            metric_name="rmse",
            metric_direction="minimize",
        )

        # 1. Valid submission
        valid_path = self.work_dir / "valid.csv"
        pd.DataFrame({"id": [10, 20, 30], "site_eui": [1.5, 2.5, 3.5]}).to_csv(
            valid_path, index=False
        )
        ok, msg = validate_submission(valid_path, spec, self.test_path)
        self.assertTrue(ok, msg)

        # 2. Missing row
        bad_rows = self.work_dir / "bad_rows.csv"
        pd.DataFrame({"id": [10, 20], "site_eui": [1.5, 2.5]}).to_csv(
            bad_rows, index=False
        )
        ok, msg = validate_submission(bad_rows, spec, self.test_path)
        self.assertFalse(ok)
        self.assertIn("row count", msg.lower())

        # 3. Wrong ID order
        bad_order = self.work_dir / "bad_order.csv"
        pd.DataFrame({"id": [20, 10, 30], "site_eui": [1.5, 2.5, 3.5]}).to_csv(
            bad_order, index=False
        )
        ok, msg = validate_submission(bad_order, spec, self.test_path)
        self.assertFalse(ok)
        self.assertIn("id mismatch", msg.lower())

        # 4. NaN / Inf
        bad_nan = self.work_dir / "bad_nan.csv"
        pd.DataFrame({"id": [10, 20, 30], "site_eui": [1.5, np.nan, 3.5]}).to_csv(
            bad_nan, index=False
        )
        ok, msg = validate_submission(bad_nan, spec, self.test_path)
        self.assertFalse(ok)
        self.assertIn("nan", msg.lower())

    def test_classification_class_validation(self):
        from experiments import TaskSpec
        from submission import validate_submission

        spec = TaskSpec(
            problem_type="multiclass",
            id_column="id",
            target_column="attack_cat",
            metric_name="macro_f1",
            metric_direction="maximize",
            classes=["Normal", "Generic", "Exploits"],
        )
        # Invalid class name
        bad_cls = self.work_dir / "bad_cls.csv"
        pd.DataFrame({"id": [10, 20, 30], "attack_cat": ["Normal", "AlienAttack", "Exploits"]}).to_csv(
            bad_cls, index=False
        )
        ok, msg = validate_submission(bad_cls, spec, self.test_path)
        self.assertFalse(ok)
        self.assertIn("invalid class", msg.lower())

    def test_atomic_publish_and_preservation(self):
        from submission import atomic_publish
        target = self.work_dir / "submission.csv"
        src1 = self.work_dir / "src1.csv"
        src1.write_text("id,val\n1,10\n", encoding="utf-8")
        
        atomic_publish(src1, target)
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "id,val\n1,10\n")

        # Second publish replaces atomically
        src2 = self.work_dir / "src2.csv"
        src2.write_text("id,val\n1,20\n", encoding="utf-8")
        atomic_publish(src2, target)
        self.assertEqual(target.read_text(encoding="utf-8"), "id,val\n1,20\n")

    def test_reserve_created_without_fit(self):
        from experiments import TaskSpec
        from submission import create_reserve_submission, validate_submission

        spec = TaskSpec(
            problem_type="regression",
            id_column="id",
            target_column="site_eui",
            metric_name="rmse",
            metric_direction="minimize",
        )
        sample = self.work_dir / "sample_submission.csv"
        pd.DataFrame({"id": [10, 20, 30], "site_eui": [0.0, 0.0, 0.0]}).to_csv(
            sample, index=False
        )
        reserve_path = self.work_dir / "reserve.csv"
        create_reserve_submission(spec, self.test_path, sample, reserve_path)
        
        self.assertTrue(reserve_path.exists())
        ok, msg = validate_submission(reserve_path, spec, self.test_path)
        self.assertTrue(ok, msg)


class TestExperimentsValidationAndMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_confirmation_split_no_leak(self):
        from experiments import split_data
        n_samples = 100
        df = pd.DataFrame({"x": range(n_samples), "y": np.random.randn(n_samples)})
        split = split_data(df, target_col="y", problem_type="regression", confirmation_ratio=0.2, val_ratio=0.2, seed=42)
        
        # Check all sets are disjoint
        train_idx = set(split.train_indices)
        val_idx = set(split.val_indices)
        conf_idx = set(split.confirmation_indices)
        
        self.assertEqual(len(train_idx.intersection(val_idx)), 0)
        self.assertEqual(len(train_idx.intersection(conf_idx)), 0)
        self.assertEqual(len(val_idx.intersection(conf_idx)), 0)
        self.assertEqual(len(train_idx) + len(val_idx) + len(conf_idx), n_samples)

    def test_unknown_metric_rejection(self):
        from experiments import resolve_metric
        # Known metrics
        m1, d1 = resolve_metric("rmse")
        self.assertEqual(d1, "minimize")
        m2, d2 = resolve_metric("macro_f1")
        self.assertEqual(d2, "maximize")

        # Unknown metric must raise ValueError without inventing a proxy
        with self.assertRaises(ValueError):
            resolve_metric("unsupported_custom_metric_xyz")

    def test_trusted_evaluator_exact_metrics(self):
        from experiments import compute_metric
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        rmse = compute_metric("rmse", y_true, y_pred)
        # diffs: 0, 0, 2 -> mse = 4/3 -> rmse = sqrt(4/3)
        self.assertAlmostEqual(rmse, (4.0 / 3.0) ** 0.5)

        y_true_cls = ["A", "B", "A", "B"]
        y_pred_cls = ["A", "B", "B", "B"]
        f1 = compute_metric("macro_f1", y_true_cls, y_pred_cls)
        self.assertGreater(f1, 0.0)
        self.assertLessEqual(f1, 1.0)


class TestAgentAndJournal(unittest.TestCase):
    def test_action_validation(self):
        from agent import validate_action
        valid_action = {
            "name": "propose_recipe",
            "arguments": {
                "hypothesis": "Use lightgbm baseline",
                "recipe": {
                    "model_type": "lightgbm",
                    "params": {"learning_rate": 0.05, "n_estimators": 50},
                    "scaler": "standard",
                    "encoder": "onehot",
                },
            },
        }
        ok, msg = validate_action(valid_action)
        self.assertTrue(ok, msg)

        # Invalid: prohibited model or unknown tool
        invalid_tool = {"name": "run_bash_command", "arguments": {"cmd": "rm -rf"}}
        ok, msg = validate_action(invalid_tool)
        self.assertFalse(ok)

        invalid_model = {
            "name": "propose_recipe",
            "arguments": {"hypothesis": "x", "recipe": {"model_type": "gpt4_dnn"}},
        }
        ok, msg = validate_action(invalid_model)
        self.assertFalse(ok)

    def test_journal_modes(self):
        from agent import Journal
        journal = Journal()
        self.assertEqual(journal.next_mode(), "draft")
        
        # Add failed experiment
        journal.record_node(
            node_id=1,
            parent_id=0,
            hypothesis="First draft",
            recipe={"model_type": "linear"},
            status="error",
            error="CUDA not found",
        )
        self.assertEqual(journal.next_mode(), "debug")

        # Add successful experiment
        journal.record_node(
            node_id=2,
            parent_id=1,
            hypothesis="Fix bug",
            recipe={"model_type": "lightgbm"},
            status="success",
            metric=0.45,
            csv_path="cand2.csv",
        )
        self.assertEqual(journal.next_mode(), "improve")
        self.assertEqual(journal.get_best_node().id, 2)


class TestFullMockQwenCycle(unittest.TestCase):
    def test_cycle_with_mock_qwen(self):
        from agent import LocalAgentController
        from experiments import TaskSpec

        spec = TaskSpec(
            problem_type="regression",
            id_column="id",
            target_column="target",
            metric_name="rmse",
            metric_direction="minimize",
        )
        controller = LocalAgentController()
        # Generates sequence of deterministic hypotheses
        a1 = controller.propose_next_action([], spec)
        self.assertEqual(a1["name"], "propose_recipe")
        self.assertIn("recipe", a1["arguments"])
        
        a2 = controller.propose_next_action([{"status": "success", "metric": 10.0}], spec)
        self.assertEqual(a2["name"], "propose_recipe")


if __name__ == "__main__":
    unittest.main()
