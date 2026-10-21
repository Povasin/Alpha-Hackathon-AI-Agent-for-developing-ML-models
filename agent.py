"""AutoResearch agent, experiment journal, Qwen client adapter, and local controller."""

from dataclasses import dataclass, field
import json
import os
from typing import Any

from experiments import TaskSpec

ALLOWED_MODELS = {"lightgbm", "catboost", "linear", "dummy", "random_forest"}
ALLOWED_SCALERS = {"none", "standard", "robust"}
ALLOWED_ENCODERS = {"onehot", "ordinal", "target"}
ALLOWED_TOOLS = {"propose_recipe", "stop_search"}


def validate_action(action: dict[str, Any]) -> tuple[bool, str]:
    """Validate action structure and permissible recipe grammar."""
    if not isinstance(action, dict):
        return False, "Action must be a JSON object"

    name = action.get("name")
    if name not in ALLOWED_TOOLS:
        return False, f"Unknown action tool '{name}'. Allowed tools: {sorted(ALLOWED_TOOLS)}"

    if name == "stop_search":
        return True, "Valid stop_search action"

    args = action.get("arguments", {})
    if not isinstance(args, dict):
        return False, "Arguments must be a JSON object"

    hypothesis = args.get("hypothesis")
    if not hypothesis or not isinstance(hypothesis, str):
        return False, "Missing or invalid 'hypothesis' in arguments"

    recipe = args.get("recipe")
    if not isinstance(recipe, dict):
        return False, "Missing or invalid 'recipe' in arguments"

    model_type = recipe.get("model_type")
    if model_type not in ALLOWED_MODELS:
        return (
            False,
            f"Disallowed model_type '{model_type}'. Allowed: {sorted(ALLOWED_MODELS)}",
        )

    scaler = recipe.get("scaler", "standard")
    if scaler not in ALLOWED_SCALERS:
        return False, f"Disallowed scaler '{scaler}'. Allowed: {sorted(ALLOWED_SCALERS)}"

    encoder = recipe.get("encoder", "onehot")
    if encoder not in ALLOWED_ENCODERS:
        return False, f"Disallowed encoder '{encoder}'. Allowed: {sorted(ALLOWED_ENCODERS)}"

    return True, "Valid action"


@dataclass
class ExperimentNode:
    id: int
    parent_id: int
    hypothesis: str
    recipe: dict[str, Any]
    status: str  # "success" | "error" | "pending"
    metric: float | None = None
    error: str | None = None
    csv_path: str | None = None
    duration: float = 0.0


class Journal:
    """Tree of AutoResearch experiments tracking history, hypotheses, and scores."""

    def __init__(self, metric_direction: str = "minimize"):
        self.metric_direction = metric_direction
        self.nodes: list[ExperimentNode] = []
        self._next_id = 1

    def record_node(
        self,
        node_id: int | None = None,
        parent_id: int = 0,
        hypothesis: str = "",
        recipe: dict[str, Any] | None = None,
        status: str = "success",
        metric: float | None = None,
        error: str | None = None,
        csv_path: str | None = None,
        duration: float = 0.0,
    ) -> ExperimentNode:
        """Add experiment trial to journal."""
        if node_id is None:
            node_id = self._next_id
            self._next_id += 1
        else:
            self._next_id = max(self._next_id, node_id + 1)

        node = ExperimentNode(
            id=node_id,
            parent_id=parent_id,
            hypothesis=hypothesis,
            recipe=recipe or {},
            status=status,
            metric=metric,
            error=error,
            csv_path=csv_path,
            duration=duration,
        )
        self.nodes.append(node)
        return node

    def next_mode(self) -> str:
        """Determine next agent loop mode: draft, debug, or improve."""
        if not self.nodes:
            return "draft"

        last = self.nodes[-1]
        if last.status == "error":
            return "debug"

        successful = [n for n in self.nodes if n.status == "success" and n.metric is not None]
        if not successful:
            return "draft"

        return "improve"

    def get_best_node(self) -> ExperimentNode | None:
        """Find node with best validation metric."""
        successful = [n for n in self.nodes if n.status == "success" and n.metric is not None]
        if not successful:
            return None

        if self.metric_direction == "minimize":
            return min(successful, key=lambda n: n.metric)  # type: ignore
        return max(successful, key=lambda n: n.metric)  # type: ignore

    def get_shortlist(self, n: int = 3) -> list[ExperimentNode]:
        """Return top N distinct successful candidates."""
        successful = [n for n in self.nodes if n.status == "success" and n.metric is not None]
        reverse = self.metric_direction == "maximize"
        sorted_nodes = sorted(successful, key=lambda x: x.metric, reverse=reverse)  # type: ignore
        return sorted_nodes[:n]


class LocalAgentController:
    """Deterministic fallback controller when Qwen API is unavailable."""

    def __init__(self):
        self.step_counter = 0

    def propose_next_action(
        self, history: list[dict[str, Any]], task_spec: TaskSpec
    ) -> dict[str, Any]:
        """Return next exploration recipe deterministically based on iteration step."""
        self.step_counter += 1
        idx = self.step_counter

        if idx == 1:
            return {
                "name": "propose_recipe",
                "arguments": {
                    "hypothesis": "Draft: Simple baseline linear/ridge model",
                    "recipe": {
                        "model_type": "linear",
                        "scaler": "standard",
                        "encoder": "onehot",
                    },
                },
            }
        elif idx == 2:
            return {
                "name": "propose_recipe",
                "arguments": {
                    "hypothesis": "Improve: LightGBM gradient boosting with default params",
                    "recipe": {
                        "model_type": "lightgbm",
                        "params": {"learning_rate": 0.05, "n_estimators": 100},
                        "scaler": "standard",
                        "encoder": "onehot",
                    },
                },
            }
        elif idx == 3:
            return {
                "name": "propose_recipe",
                "arguments": {
                    "hypothesis": "Improve: CatBoost gradient boosting baseline",
                    "recipe": {
                        "model_type": "catboost",
                        "params": {"learning_rate": 0.05, "iterations": 100},
                        "scaler": "robust",
                        "encoder": "onehot",
                    },
                },
            }
        elif idx == 4:
            return {
                "name": "propose_recipe",
                "arguments": {
                    "hypothesis": "Improve: LightGBM with robust scaling and higher depth",
                    "recipe": {
                        "model_type": "lightgbm",
                        "params": {
                            "learning_rate": 0.03,
                            "n_estimators": 150,
                            "max_depth": 7,
                            "num_leaves": 63,
                        },
                        "scaler": "robust",
                        "encoder": "onehot",
                    },
                },
            }
        elif idx == 5:
            return {
                "name": "propose_recipe",
                "arguments": {
                    "hypothesis": "Improve: CatBoost with tuned regularization and depth",
                    "recipe": {
                        "model_type": "catboost",
                        "params": {"learning_rate": 0.03, "iterations": 150, "depth": 6},
                        "scaler": "standard",
                        "encoder": "onehot",
                    },
                },
            }
        else:
            return {
                "name": "stop_search",
                "arguments": {"reason": "Exploration search limit reached"},
            }


class QwenAgent:
    """Agent interface communicating with Qwen model via OpenAI-compatible endpoint."""

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        self.is_available = bool(self.api_key)
        self.fallback = LocalAgentController()

    def get_system_prompt(self, task_spec: TaskSpec) -> str:
        return f"""You are an expert AutoML research agent optimizing ML models.
Problem: {task_spec.problem_type}
Target: {task_spec.target_column}
Metric: {task_spec.metric_name} (direction: {task_spec.metric_direction})

You must respond with a JSON object calling one of the allowed actions:
1. "propose_recipe": {{"name": "propose_recipe", "arguments": {{"hypothesis": "...", "recipe": {{"model_type": "lightgbm"|"catboost"|"linear", "scaler": "standard"|"robust"|"none", "encoder": "onehot", "params": {{...}}}}}}}}
2. "stop_search": {{"name": "stop_search", "arguments": {{"reason": "..."}}}}

Do not output code or markdown outside of the JSON object.
"""

    def propose_action(
        self, journal: Journal, task_spec: TaskSpec
    ) -> tuple[dict[str, Any], str]:
        """Propose next action using Qwen if available, otherwise fallback."""
        if not self.is_available:
            action = self.fallback.propose_next_action(
                [{"status": n.status, "metric": n.metric} for n in journal.nodes],
                task_spec,
            )
            return action, "local_fallback"

        # Qwen API call via openai library
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
                timeout=30.0,
            )
            mode = journal.next_mode()
            best_node = journal.get_best_node()
            best_info = (
                f"Best metric so far: {best_node.metric} (node {best_node.id})"
                if best_node
                else "No successful runs yet"
            )

            prompt = f"Mode: {mode}\n{best_info}\nPropose next experiment action."
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_system_prompt(task_spec)},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            action = json.loads(content)
            ok, msg = validate_action(action)
            if ok:
                return action, "qwen"
            # If invalid action, fallback to safe local controller
            return (
                self.fallback.propose_next_action(
                    [{"status": n.status, "metric": n.metric} for n in journal.nodes],
                    task_spec,
                ),
                f"qwen_invalid_fallback ({msg})",
            )
        except Exception as exc:
            return (
                self.fallback.propose_next_action(
                    [{"status": n.status, "metric": n.metric} for n in journal.nodes],
                    task_spec,
                ),
                f"qwen_error_fallback ({exc})",
            )
