"""Task specification, validation protocols, trusted evaluation, and recipe compilation."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from runtime import run_subprocess


@dataclass
class TaskSpec:
    problem_type: str  # "regression" | "multiclass" | "binary"
    id_column: str
    target_column: str
    metric_name: str
    metric_direction: str  # "minimize" | "maximize"
    classes: list[str] | None = None
    train_path: Path | None = None
    test_path: Path | None = None
    sample_path: Path | None = None
    dictionary_path: Path | None = None


@dataclass
class ValidationSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray
    confirmation_indices: np.ndarray


TRUSTED_METRICS = {
    "rmse": ("rmse", "minimize"),
    "macro_f1": ("macro_f1", "maximize"),
    "accuracy": ("accuracy", "maximize"),
    "mae": ("mae", "minimize"),
}


def resolve_metric(metric_raw: str) -> tuple[str, str]:
    """Map raw metric name to trusted canonical metric or raise ValueError."""
    cleaned = metric_raw.lower().replace(" ", "_").replace("-", "_")
    if "macro" in cleaned and "f1" in cleaned:
        return TRUSTED_METRICS["macro_f1"]
    if "rmse" in cleaned or "root_mean_squared" in cleaned:
        return TRUSTED_METRICS["rmse"]
    if "accuracy" in cleaned:
        return TRUSTED_METRICS["accuracy"]
    if "mae" in cleaned or "mean_absolute" in cleaned:
        return TRUSTED_METRICS["mae"]
    if cleaned in TRUSTED_METRICS:
        return TRUSTED_METRICS[cleaned]
    raise ValueError(f"Unknown metric '{metric_raw}'. Proxy metrics are strictly prohibited.")


def compute_metric(metric_name: str, y_true: Any, y_pred: Any) -> float:
    """Compute trusted metric without relying on external outputs."""
    metric, _ = resolve_metric(metric_name)
    if metric == "rmse":
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(mean_squared_error(y_t, y_p)))
    elif metric == "macro_f1":
        y_t = [str(x) for x in y_true]
        y_p = [str(x) for x in y_pred]
        return float(f1_score(y_t, y_p, average="macro", zero_division=0))
    elif metric == "accuracy":
        y_t = [str(x) for x in y_true]
        y_p = [str(x) for x in y_pred]
        return float(accuracy_score(y_t, y_p))
    elif metric == "mae":
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        return float(mean_absolute_error(y_t, y_p))
    raise ValueError(f"No evaluator for metric '{metric}'")


def split_data(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    confirmation_ratio: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> ValidationSplit:
    """Split dataset into disjoint train, val, and untouched confirmation sets."""
    n_samples = len(df)
    indices = np.arange(n_samples)
    y = df[target_col].values

    # Step 1: Separate confirmation set
    if problem_type in ("multiclass", "binary"):
        try:
            splitter1 = StratifiedShuffleSplit(
                n_splits=1, test_size=confirmation_ratio, random_state=seed
            )
            search_idx, conf_idx = next(splitter1.split(indices, y))
        except Exception:
            splitter1 = ShuffleSplit(
                n_splits=1, test_size=confirmation_ratio, random_state=seed
            )
            search_idx, conf_idx = next(splitter1.split(indices))
    else:
        splitter1 = ShuffleSplit(
            n_splits=1, test_size=confirmation_ratio, random_state=seed
        )
        search_idx, conf_idx = next(splitter1.split(indices))

    # Step 2: Split search data into train and val
    search_y = y[search_idx]
    relative_val = val_ratio / (1.0 - confirmation_ratio)
    if problem_type in ("multiclass", "binary"):
        try:
            splitter2 = StratifiedShuffleSplit(
                n_splits=1, test_size=relative_val, random_state=seed + 1
            )
            train_sub, val_sub = next(splitter2.split(search_idx, search_y))
        except Exception:
            splitter2 = ShuffleSplit(
                n_splits=1, test_size=relative_val, random_state=seed + 1
            )
            train_sub, val_sub = next(splitter2.split(search_idx))
    else:
        splitter2 = ShuffleSplit(
            n_splits=1, test_size=relative_val, random_state=seed + 1
        )
        train_sub, val_sub = next(splitter2.split(search_idx))

    return ValidationSplit(
        train_indices=search_idx[train_sub],
        val_indices=search_idx[val_sub],
        confirmation_indices=conf_idx,
    )


def resolve_task_spec(task_path: Path, work_dir: Path) -> TaskSpec:
    """Resolve TaskSpec from task.md, data directory, and CSV headers."""
    task_text = task_path.read_text(encoding="utf-8") if task_path.exists() else ""
    task_dir = task_path.parent

    # Locate CSV files
    def find_csv(name: str) -> Path:
        # 1. Check path explicitly mentioned in task.md
        for match in re.finditer(rf'[\s`\'"]([^\s`\'"]*?{re.escape(name)})', task_text):
            p = task_dir / match.group(1)
            if p.exists():
                return p
        # 2. Check directly in task_dir or work_dir
        for p in [task_dir / name, work_dir / name]:
            if p.exists():
                return p
        # 3. Recursive glob
        matches = list(task_dir.glob(f"**/{name}"))
        if matches:
            return matches[0]
        matches_work = list(work_dir.glob(f"**/{name}"))
        if matches_work:
            return matches_work[0]
        return task_dir / name

    sample_path = find_csv("sample_submission.csv")
    train_path = find_csv("train.csv")
    test_path = find_csv("test.csv")
    dict_path = find_csv("data_dictionary.csv")

    # Read sample submission to determine ID and target column
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample_submission.csv at {sample_path}")
    sample_df = pd.read_csv(sample_path, nrows=5)
    cols = list(sample_df.columns)
    if len(cols) != 2:
        raise ValueError(f"Sample submission must have exactly 2 columns, got {cols}")
    id_col, target_col = cols[0], cols[1]

    # Read small sample of train to determine problem type & classes
    train_sample = pd.read_csv(train_path, nrows=100)
    target_series = train_sample[target_col]

    classes = None
    if re.search(r"macro[_\s-]*f1", task_text, re.IGNORECASE) or not pd.api.types.is_numeric_dtype(
        target_series.dtype
    ):
        problem_type = "multiclass"
        metric_name, metric_dir = resolve_metric("macro_f1")
        # Read full target column to find exact classes
        full_target = pd.read_csv(train_path, usecols=[target_col])[target_col]
        classes = sorted(list(full_target.astype(str).unique()))
    else:
        problem_type = "regression"
        metric_name, metric_dir = resolve_metric("rmse")

    return TaskSpec(
        problem_type=problem_type,
        id_column=id_col,
        target_column=target_col,
        metric_name=metric_name,
        metric_direction=metric_dir,
        classes=classes,
        train_path=train_path,
        test_path=test_path,
        sample_path=sample_path,
        dictionary_path=dict_path,
    )


def compile_recipe_code(recipe: dict, task_spec: TaskSpec) -> str:
    """Generate self-contained Python code for trainer script."""
    recipe_json = json.dumps(recipe)
    spec_json = json.dumps(
        {
            "problem_type": task_spec.problem_type,
            "id_column": task_spec.id_column,
            "target_column": task_spec.target_column,
            "metric_name": task_spec.metric_name,
            "classes": task_spec.classes,
        }
    )

    code = f'''# Auto-generated trainer script
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer

recipe = json.loads({repr(recipe_json)})
spec = json.loads({repr(spec_json)})

train_path = Path(sys.argv[1])
test_path = Path(sys.argv[2])
split_path = Path(sys.argv[3])
out_dir = Path(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)

with open(split_path, "r", encoding="utf-8") as f:
    split = json.load(f)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

id_col = spec["id_column"]
target_col = spec["target_column"]
problem_type = spec["problem_type"]

train_idx = split["train_indices"]
val_idx = split["val_indices"]

X_train_df = train_df.iloc[train_idx].drop(columns=[target_col, id_col], errors="ignore")
y_train = np.asarray(train_df.iloc[train_idx][target_col].values)

X_val_df = train_df.iloc[val_idx].drop(columns=[target_col, id_col], errors="ignore")
y_val = np.asarray(train_df.iloc[val_idx][target_col].values)

X_test_df = test_df.drop(columns=[id_col], errors="ignore")

# Separate numeric and categorical
num_cols = [c for c in X_train_df.columns if pd.api.types.is_numeric_dtype(X_train_df[c].dtype)]
cat_cols = [c for c in X_train_df.columns if c not in num_cols]

# Impute numeric
for c in num_cols:
    med = X_train_df[c].median()
    X_train_df[c] = X_train_df[c].fillna(med)
    X_val_df[c] = X_val_df[c].fillna(med)
    X_test_df[c] = X_test_df[c].fillna(med)

# Impute categorical
for c in cat_cols:
    mode_val = X_train_df[c].mode()[0] if not X_train_df[c].mode().empty else "missing"
    X_train_df[c] = X_train_df[c].fillna(mode_val).astype(str)
    X_val_df[c] = X_val_df[c].fillna(mode_val).astype(str)
    X_test_df[c] = X_test_df[c].fillna(mode_val).astype(str)

# Preprocessing: Encoding
encoder_type = recipe.get("encoder", "onehot")
if cat_cols:
    combined = pd.concat([X_train_df[cat_cols], X_val_df[cat_cols], X_test_df[cat_cols]], axis=0)
    dummies = pd.get_dummies(combined, dummy_na=False, sparse=False, drop_first=True)
    n1 = len(X_train_df)
    n2 = len(X_val_df)
    X_train_df = pd.concat([X_train_df[num_cols].reset_index(drop=True), dummies.iloc[:n1].reset_index(drop=True)], axis=1)
    X_val_df = pd.concat([X_val_df[num_cols].reset_index(drop=True), dummies.iloc[n1:n1+n2].reset_index(drop=True)], axis=1)
    X_test_df = pd.concat([X_test_df[num_cols].reset_index(drop=True), dummies.iloc[n1+n2:].reset_index(drop=True)], axis=1)
else:
    X_train_df = X_train_df[num_cols].reset_index(drop=True)
    X_val_df = X_val_df[num_cols].reset_index(drop=True)
    X_test_df = X_test_df[num_cols].reset_index(drop=True)

# Scaling
scaler_type = recipe.get("scaler", "standard")
if scaler_type == "standard":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)
elif scaler_type == "robust":
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)
else:
    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = X_test_df.values

# Model selection and fit
model_type = recipe.get("model_type", "lightgbm")
params = recipe.get("params", {{}})

t0 = time.time()
if problem_type == "regression":
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        p = {{"n_estimators": 100, "learning_rate": 0.05, "verbose": -1, "n_jobs": 4}}
        p.update(params)
        model = LGBMRegressor(**p)
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        p = {{"iterations": 100, "learning_rate": 0.05, "verbose": 0, "thread_count": 4}}
        p.update(params)
        model = CatBoostRegressor(**p)
    elif model_type == "linear":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    else:
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor(strategy="mean")

    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    metric_val = float(np.sqrt(mean_squared_error(y_val, val_preds)))
else:
    # Classification
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        p = {{"n_estimators": 100, "learning_rate": 0.05, "verbose": -1, "n_jobs": 4}}
        p.update(params)
        model = LGBMClassifier(**p)
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        p = {{"iterations": 100, "learning_rate": 0.05, "verbose": 0, "thread_count": 4}}
        p.update(params)
        model = CatBoostClassifier(**p)
    elif model_type == "linear":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=500, n_jobs=4)
    else:
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="most_frequent")

    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    metric_val = float(f1_score([str(x) for x in y_val], [str(x) for x in val_preds], average="macro", zero_division=0))

duration = time.time() - t0

# Output results
result = {{
    "status": "success",
    "metric": metric_val,
    "duration": duration,
}}
with open(out_dir / "eval_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f)

# Output test predictions CSV
sub_df = pd.DataFrame({{
    id_col: test_df[id_col].values,
    target_col: test_preds,
}})
sub_df.to_csv(out_dir / "test_predictions.csv", index=False)
'''
    return code
