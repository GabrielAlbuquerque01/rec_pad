import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from src.evaluation import make_splits, metrics
from src.hybrid import CompetenceRouter, estimate_competence, fuse, normalize_competence
from src.models import aligned_probabilities, make_models
from src.preprocessing import make_preprocessor
from src.statistical_tests import compare


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 0, 1])
        self.probabilities = np.array([
            [[0.8, 0.2], [0.1, 0.9]], [[0.2, 0.8], [0.1, 0.9]],
            [[0.8, 0.2], [0.9, 0.1]], [[0.2, 0.8], [0.9, 0.1]],
        ])

    def test_competence_and_hand_calculated_fusion(self):
        global_c, local_c, global_w, local_w = estimate_competence(
            self.probabilities, self.labels, np.array([0, 0, 1, 1]), 3)
        assert_allclose(global_c, [1, 0.5])
        assert_allclose(global_w, [2 / 3, 1 / 3])
        assert_allclose(local_c[:2], [[1, 0.5], [1, 0.5]])
        assert_allclose(local_w.sum(axis=1), 1)
        self.assertTrue((local_w >= 0).all())
        self.assertTrue(np.isnan(local_c[2]).all())
        assert_allclose(local_w[2], global_w)
        assert_allclose(fuse(self.probabilities, global_w)[0], [17 / 30, 13 / 30])
        assert_allclose(fuse(self.probabilities, global_w), fuse(self.probabilities, local_w[[0, 0, 1, 1]]))

    def test_one_region_and_zero_fallbacks(self):
        _, _, global_w, local_w = estimate_competence(self.probabilities, self.labels, np.zeros(4), 1)
        assert_allclose(fuse(self.probabilities, global_w), fuse(self.probabilities, local_w[np.zeros(4, dtype=int)]))
        assert_allclose(normalize_competence(np.zeros(5)), np.ones(5) / 5)
        assert_allclose(normalize_competence(np.zeros(2), np.array([0.2, 0.8])), [0.2, 0.8])

    def test_probability_alignment_and_ties(self):
        class ReverseModel:
            classes_ = np.array([1, 0])

            def predict_proba(self, _):
                return np.array([[0.1, 0.9], [0.5, 0.5]])

        aligned = aligned_probabilities(ReverseModel(), None)
        assert_allclose(aligned, [[0.9, 0.1], [0.5, 0.5]])
        assert_array_equal(aligned.argmax(axis=1), [0, 0])

    def test_distinct_local_weights_and_actual_zero_region(self):
        probabilities = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [0., 1.]]])
        # Region 0: first model right, second wrong. Region 1: both wrong.
        _, local_c, global_w, local_w = estimate_competence(probabilities, np.array([0, 0]), np.array([0, 1]), 2)
        assert_allclose(local_c, [[1, 0], [0, 0]])
        assert_allclose(local_w[0], [1, 0])
        assert_allclose(local_w[1], global_w)
        # Opposite competent models in different regions produce distinct fusion.
        probabilities[1] = [[0, 1], [1, 0]]
        _, _, global_w, local_w = estimate_competence(probabilities, np.array([0, 0]), np.array([0, 1]), 2)
        assert_allclose(global_w, [.5, .5])
        assert_allclose(local_w, [[1, 0], [0, 1]])
        assert_allclose(fuse(probabilities, local_w), [[1, 0], [1, 0]])

    def test_nested_groups_and_coverage(self):
        groups = np.repeat(np.arange(60), 2)
        labels = np.repeat(np.arange(60) % 2, 2)
        coverage = np.zeros(120)
        for train, test in make_splits(labels, groups, 5):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            coverage[test] += 1
            inner_coverage = np.zeros(len(train))
            for inner_train, validation in make_splits(labels[train], groups[train], 3):
                self.assertFalse(set(groups[train[inner_train]]) & set(groups[train[validation]]))
                self.assertFalse(set(train[validation]) & set(test))
                inner_coverage[validation] += 1
            assert_array_equal(inner_coverage, 1)
        assert_array_equal(coverage, 1)

    def test_preprocessing_uses_training_only(self):
        train = np.array([[1, np.nan, 9], [3, 2, 9], [5, 4, 9]])
        processor = make_preprocessor().fit(train)
        assert_allclose(processor.named_steps["impute"].statistics_, [3, 3, 9])
        assert_allclose(processor.named_steps["scale"].mean_, [3, 3])
        processor.transform([[10000, 9000, -1]])
        assert_allclose(processor.named_steps["scale"].mean_, [3, 3])
        self.assertEqual(processor.transform(train).shape, (3, 2))

    def test_metrics_continuous_auc_and_undefined_precision(self):
        values = metrics(np.array([0, 0, 1, 1]), np.array([[.9, .1], [.8, .2], [.7, .3], [.6, .4]]))
        self.assertEqual(values["roc_auc"], 1)
        self.assertEqual(values["accuracy"], .5)
        self.assertTrue(np.isnan(values["precision"]))
        self.assertIn("precision", values["undefined"])

    def test_all_models_and_router_on_real_data(self):
        from src.datasets import load_dataset
        features, labels, groups, _ = load_dataset("sonar")
        train, test = make_splits(labels, groups, 5)[0]
        router = CompetenceRouter().fit(features.iloc[train])
        self.assertEqual(len(router.predict(features.iloc[test])), len(test))
        diagnostics = router.diagnostics()
        self.assertEqual(sum(r["size"] for r in diagnostics), len(train))
        total_ss = ((router.training_features_ - router.training_features_.mean(axis=0)) ** 2).sum()
        assert_allclose(sum(r["within_ss"] + r["between_ss"] for r in diagnostics), total_ss)
        for model in make_models().values():
            model.fit(features.iloc[train], labels[train])
            probabilities = aligned_probabilities(model, features.iloc[test])
            self.assertEqual(probabilities.shape, (len(test), 2))

    def test_incomplete_statistics_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                compare(pd.DataFrame([{"dataset": "sonar", "model": "M1", "fold": 1, "roc_auc": .7}]), Path(directory))


if __name__ == "__main__":
    unittest.main()
