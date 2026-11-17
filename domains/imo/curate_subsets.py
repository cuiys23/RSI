import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from domains.imo.grading_utils import GROUND_TRUTH_KEY, QUESTION_ID, normalize_points


def print_stats(name, df):
    print(f"== {name}: {len(df)} ==")
    print(df[GROUND_TRUTH_KEY].value_counts().sort_index().to_dict())


def curate_subsets_gradingbench(csv_path=None):
    csv_path = Path(csv_path) if csv_path else Path(__file__).with_name("gradingbench.csv")
    df = pd.read_csv(csv_path, dtype=str)
    if df[QUESTION_ID].isna().any() or df[QUESTION_ID].duplicated().any():
        raise ValueError("Grading IDs must be present and unique")
    df[GROUND_TRUTH_KEY] = df[GROUND_TRUTH_KEY].map(normalize_points)
    df = df[df[GROUND_TRUTH_KEY].notna()].copy()
    df[GROUND_TRUTH_KEY] = df[GROUND_TRUTH_KEY].astype(int)
    manifest_path = csv_path.with_name("grading_split_ids.json")
    names = ("train", "val", "test")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = manifest["splits"]
    else:
        # Stratify both selection of the 300 examples and their partition.
        if len(df) < 300:
            raise ValueError("At least 300 valid examples are required")
        subset = df if len(df) == 300 else train_test_split(
            df, train_size=300, random_state=42, stratify=df[GROUND_TRUTH_KEY]
        )[0]
        train, temp = train_test_split(
            subset, train_size=100, random_state=42, stratify=subset[GROUND_TRUTH_KEY]
        )
        val, test = train_test_split(
            temp, train_size=100, random_state=42, stratify=temp[GROUND_TRUTH_KEY]
        )
        ids = {name: part[QUESTION_ID].tolist()
               for name, part in zip(names, (train, val, test))}
        manifest = {"dataset": "IMO-GradingBench", "seed": 42,
                    "label_column": GROUND_TRUTH_KEY, "splits": ids}
    all_ids = [qid for name in names for qid in ids[name]]
    if any(len(ids[name]) != 100 for name in names) or len(set(all_ids)) != 300:
        raise ValueError("Splits must contain 100 unique, disjoint IDs each")
    indexed = df.set_index(QUESTION_ID, drop=False)
    if not set(all_ids).issubset(indexed.index):
        raise ValueError("Saved split IDs are missing or have invalid Points in this dataset")
    # Preserve both membership and order when reusing the manifest.
    for name in names:
        part = indexed.loc[ids[name]]
        part.to_csv(csv_path.with_name(f"gradingbench_filtered_100_{name}.csv"), index=False)
        print_stats(name, part)
    df.sample(frac=1, random_state=42).to_csv(
        csv_path.with_name("gradingbench_filtered.csv"), index=False
    )
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    curate_subsets_gradingbench()
