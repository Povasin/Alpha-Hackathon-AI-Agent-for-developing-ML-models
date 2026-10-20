"""Read-only profiling of supplied hackathon CSVs; no model training or API calls."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def profile(name, target):
    folder = ROOT / "data" / name
    train = pd.read_csv(folder / "train.csv")
    test = pd.read_csv(folder / "test.csv")
    features = train.columns.drop(["id", target]).tolist()
    x = train[features]
    # Exact grouping, including missing values; no assumption that id is an entity.
    groups = train.groupby(features, dropna=False, observed=True)[target].agg(
        size="size", distinct_targets="nunique"
    )
    fingerprints = pd.util.hash_pandas_object(x, index=False)
    overlap = pd.util.hash_pandas_object(test[features], index=False).isin(fingerprints)
    cats = x.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    result = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "feature_count": len(features),
        "categorical_cardinality": {c: int(train[c].nunique()) for c in cats},
        "missing_counts": {c: int(v) for c, v in train.isna().sum().items() if v},
        "duplicate_feature_rows_beyond_first": int(x.duplicated().sum()),
        "duplicate_feature_groups": int((groups["size"] > 1).sum()),
        "groups_with_distinct_targets": int((groups.distinct_targets > 1).sum()),
        "rows_in_groups_with_distinct_targets": int(groups.loc[groups.distinct_targets > 1, "size"].sum()),
        "test_rows_with_train_feature_fingerprint": int(overlap.sum()),
        "fingerprint_method": "pandas hash_pandas_object, index=False; overlap is a fingerprint check",
        "schema_dictionary": pd.read_csv(folder / "data_dictionary.csv").columns.tolist(),
        "files_sha256": {f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
                         for f in ["train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"]},
    }
    if target == "site_eui":
        result.update({
            "target_summary": train[target].describe(percentiles=[.01, .5, .95, .99]).to_dict(),
            "years_train": train.Year_Factor.value_counts().sort_index().to_dict(),
            "years_test": test.Year_Factor.value_counts().sort_index().to_dict(),
            "year_built_zero": int(train.year_built.eq(0).sum()),
        })
    else:
        result.update({
            "class_counts": train[target].value_counts().to_dict(),
            "class_percent": (train[target].value_counts(normalize=True) * 100).round(4).to_dict(),
            "unseen_test_categories": {c: sorted(set(test[c].dropna().astype(str)) - set(train[c].dropna().astype(str))) for c in cats},
            "label_present": "label" in train.columns,
        })
    return result


if __name__ == "__main__":
    result = {"analysis_date": "2026-10-20", "data_modified": False,
              "regression": profile("regression", "site_eui"),
              "classification": profile("classifiaction", "attack_cat")}
    path = OUT / "data_profile.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    print(path)
    print(json.dumps({k: {f: result[k][f] for f in ["train_shape", "test_shape", "duplicate_feature_rows_beyond_first", "groups_with_distinct_targets"]} for k in ["regression", "classification"]}, ensure_ascii=False))
