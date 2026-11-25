"""The prescribed experiment; acquisition and reporting are operational stages."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import DATASETS, REPETITION_SEEDS
from src.datasets import acquire, audit_all, load_dataset
from src.evaluation import evaluate_dataset
from src.presentation import present
from src.statistical_tests import aggregate, compare

TABLES = ["weights", "clusters", "selection", "diagnostics"]


def source_hashes():
    # Line endings are normalized so the hash does not depend on the Git checkout settings.
    return {path.as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in [Path("main.py"), *sorted(Path("src").glob("*.py"))]}


def save_tables(collected, output):
    pd.DataFrame(collected["metrics"]).to_csv(output / "metrics_by_fold.csv", index=False)
    for table in TABLES:
        pd.DataFrame(collected[table]).to_csv(output / f"{table}.csv", index=False)
    (output / "warnings.json").write_text(json.dumps(collected["warnings"], indent=2), encoding="utf-8")


def run(output):
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ["numpy", "pandas", "scikit-learn", "scipy", "scikit-posthocs", "matplotlib", "ucimlrepo"]},
                   "repetition_seeds": list(REPETITION_SEEDS), "source_sha256": source_hashes()}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    collected = {key: [] for key in ["metrics", *TABLES, "warnings"]}
    with threadpool_limits(limits=1):
        for name in DATASETS:
            features, labels, groups, _ = load_dataset(name)
            for repetition, seed in enumerate(REPETITION_SEEDS, 1):
                result = evaluate_dataset(name, features, labels, groups, seed, repetition, output,
                                          plot=repetition == 1)
                for key, values in result.items():
                    collected[key].extend(values)
            save_tables(collected, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild reports from the saved fold records")
    args = parser.parse_args()
    output = Path("results")
    output.mkdir(exist_ok=True)
    if not args.report_only:
        acquire()
        audit_all()
        if args.acquire_only:
            return
        run(output)
    frame = pd.read_csv(output / "metrics_by_fold.csv")
    summary = aggregate(frame, output)
    statistics = compare(frame, output)
    present(frame, summary, statistics, output)
    print(json.dumps(statistics["friedman"], indent=2), flush=True)


if __name__ == "__main__":
    main()
