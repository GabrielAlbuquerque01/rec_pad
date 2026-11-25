"""UCI acquisition and conservative duplicate grouping; no learned cleaning here."""
import hashlib
import json
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from ucimlrepo.fetch import DatasetNotFoundError

from .config import DATASETS


def acquire(root=Path("data/raw")):
    root.mkdir(parents=True, exist_ok=True)
    for name, (uci_id, *_rest) in DATASETS.items():
        destination = root / name
        if (destination / "metadata.json").exists():
            continue
        print(f"Downloading {name} (UCI {uci_id})", flush=True)
        try:
            dataset = fetch_ucirepo(id=uci_id)
        except DatasetNotFoundError:
            if name not in {"climate", "qsar"}:
                raise
            acquire_official_archive(name, uci_id, destination)
            continue
        destination.mkdir(exist_ok=True)
        dataset.data.original.to_csv(destination / "original.csv", index=False)
        dataset.data.features.to_csv(destination / "features.csv", index=False)
        dataset.data.targets.to_csv(destination / "target.csv", index=False)
        dataset.variables.to_csv(destination / "variables.csv", index=False)
        (destination / "metadata.json").write_text(
            json.dumps(dataset.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def acquire_official_archive(name, uci_id, destination):
    """The same UCI datasets, when not exposed by the Python import endpoint."""
    slugs = {"climate": "climate+model+simulation+crashes", "qsar": "qsar+biodegradation"}
    url = f"https://archive.ics.uci.edu/static/public/{uci_id}/{slugs[name]}.zip"
    payload = urllib.request.urlopen(url, timeout=120).read()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    destination.mkdir(exist_ok=True)
    (destination / "source.zip").write_bytes(payload)
    if name == "climate":
        member = next(p for p in archive.namelist() if p.endswith("pop_failures.dat"))
        original = pd.read_csv(io.BytesIO(archive.read(member)), sep=r"\s+")
        features = original.iloc[:, 2:20]
        target = original.iloc[:, 20:21]
        roles = ["ID", "ID"] + ["Feature"] * 18 + ["Target"]
    else:
        member = next(p for p in archive.namelist() if p.endswith("biodeg.csv"))
        original = pd.read_csv(io.BytesIO(archive.read(member)), sep=";", header=None)
        original.columns = [f"descriptor_{i}" for i in range(1, 42)] + ["class"]
        features, target = original.iloc[:, :41], original.iloc[:, 41:]
        roles = ["Feature"] * 41 + ["Target"]
    metadata = json.loads(urllib.request.urlopen(f"https://archive.ics.uci.edu/api/dataset?id={uci_id}", timeout=120).read())
    metadata["acquisition"] = {"archive_url": url, "member": member, "reason": "Unavailable through ucimlrepo import endpoint"}
    original.to_csv(destination / "original.csv", index=False)
    features.to_csv(destination / "features.csv", index=False)
    target.to_csv(destination / "target.csv", index=False)
    pd.DataFrame({"name": original.columns, "role": roles}).to_csv(destination / "variables.csv", index=False)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def load_dataset(name, root=Path("data/raw")):
    path = root / name
    uci_id, expected_n, expected_p, mapping = DATASETS[name]
    features = pd.read_csv(path / "features.csv")
    original = pd.read_csv(path / "original.csv")
    variables = pd.read_csv(path / "variables.csv")
    removed = variables.loc[variables.role.str.lower() == "id", "name"].tolist()
    if name == "climate":
        removed += [c for c in features if c.lower() in {"study", "run", "study_id", "simulation", "simulation_id"}]
    features = features.drop(columns=removed, errors="ignore")
    features = features.apply(pd.to_numeric, errors="raise")
    if np.isinf(features.to_numpy()).any():
        raise ValueError(f"{name}: infinite attributes need investigation")
    raw_target = pd.read_csv(path / "target.csv").iloc[:, 0]
    labels = raw_target.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    target = labels.map(mapping)
    if target.isna().any() or set(target) != {0, 1}:
        raise ValueError(f"{name}: unexpected labels {labels.unique()}")
    if features.shape != (expected_n, expected_p):
        raise ValueError(f"{name}: shape {features.shape}, expected {(expected_n, expected_p)}")
    # Exact feature equality, independent of labels. Preserve conflicting labels.
    groups = features.groupby(list(features.columns), dropna=False, sort=False).ngroup().to_numpy()
    conflicts = pd.DataFrame({"group": groups, "y": target}).groupby("group").y.nunique()
    audit = {
        "dataset": name, "uci_id": uci_id, "source": f"https://archive.ics.uci.edu/dataset/{uci_id}",
        "samples": len(features), "attributes": features.shape[1],
        "original_columns": original.shape[1], "removed_identifiers": json.dumps(sorted(set(removed))),
        "missing_cells": int(features.isna().sum().sum()),
        "constant_columns": json.dumps(features.columns[features.nunique() <= 1].tolist()),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "conflicting_groups": int((conflicts > 1).sum()),
        "class_0": int((target == 0).sum()), "class_1": int((target == 1).sum()),
        "label_mapping": json.dumps(mapping),
        "splitter": "StratifiedGroupKFold" if len(set(groups)) < len(groups) else "StratifiedKFold",
        "sha256": hashlib.sha256((path / "original.csv").read_bytes()).hexdigest(),
    }
    return features, target.to_numpy(dtype=int), groups, audit


def audit_all():
    records = [load_dataset(name)[3] for name in DATASETS]
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv("results/datasets.csv", index=False)
    return records
