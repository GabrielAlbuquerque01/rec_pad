"""The single prescribed experiment; acquisition and reporting are operational stages."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.config import DATASETS
from src.datasets import acquire, audit_all, load_dataset
from src.evaluation import aggregate, evaluate_dataset
from src.presentation import present
from src.statistical_tests import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild reports from all 350 existing records")
    args = parser.parse_args()
    output = Path("results")
    output.mkdir(exist_ok=True)
    if args.report_only:
        import pandas as pd
        frame = pd.read_csv(output / "metrics_by_fold.csv")
        frame, summary = aggregate(frame.to_dict("records"))
    else:
        acquire()
        audit_all()
        if args.acquire_only:
            return
        environment = {"python": platform.python_version(), "platform": platform.platform(),
                       "packages": {name: importlib.metadata.version(name) for name in
                                    ["numpy", "pandas", "scikit-learn", "scipy", "scikit-posthocs", "matplotlib", "ucimlrepo"]},
                       "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in [Path("main.py"), *sorted(Path("src").glob("*.py"))]}}
        (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        records = []
        with threadpool_limits(limits=1):
            for name in DATASETS:
                features, labels, groups, _ = load_dataset(name)
                records.extend(evaluate_dataset(name, features, labels, groups))
                aggregate(records)
        frame, summary = aggregate(records)
    matrix, ranks, statistics = compare(frame)
    present(frame, summary, matrix, ranks, statistics)
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    main()
