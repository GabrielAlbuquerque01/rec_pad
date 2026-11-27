"""Regression checks: altered statistical conclusions must fail the artifact audit."""
import copy
import json
import unittest
from pathlib import Path

import pandas as pd

from scripts.validate_results import check_pairwise


ROOT = Path(__file__).resolve().parents[1] / "results"


class ArtifactInferenceTests(unittest.TestCase):
    def setUp(self):
        self.statistics = json.loads((ROOT / "statistics.json").read_text(encoding="utf-8"))
        self.matrix = pd.read_csv(ROOT / "auc_matrix.csv", index_col=0)

    def test_saved_pairwise_inference(self):
        check_pairwise(self.statistics, self.matrix)

    def test_tampered_adjustment_significance_and_win_counts_rejected(self):
        for field, value in (("wilcoxon_p_holm", 0.123456), ("significant", True), ("wins", 100)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.statistics)
                changed["pairwise"][0][field] = value
                with self.assertRaises(AssertionError):
                    check_pairwise(changed, self.matrix)

    def test_missing_duplicated_and_misclassified_comparisons_rejected(self):
        for mutation in ("missing", "duplicate", "family"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(self.statistics)
                if mutation == "missing":
                    changed["pairwise"].pop()
                elif mutation == "duplicate":
                    changed["pairwise"][-1] = changed["pairwise"][0]
                else:
                    changed["pairwise"][0]["family"] = "ablation"
                with self.assertRaises(AssertionError):
                    check_pairwise(changed, self.matrix)
