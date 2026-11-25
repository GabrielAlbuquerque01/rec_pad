import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from src.config import BASE_MODELS, CLUSTER_CANDIDATES
from src.evaluation import diversity, make_splits, metrics
from src.hybrid import (ClusterRegions, CompetenceRouter, NeighborRegions, apply_platt, calibrate, cluster_statistics,
                        estimate_competence, excess_weights, fit_platt, fuse, global_excess_weights,
                        local_excess_weights, neighbor_statistics, select_and_fuse)
from src.models import aligned_probabilities, make_models
from src.preprocessing import make_preprocessor
from src.statistical_tests import compare, critical_difference, holm, paired_test


class OriginalHybridTests(unittest.TestCase):
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
        self.assertTrue(np.isnan(local_c[2]).all())
        assert_allclose(local_w[2], global_w)
        assert_allclose(fuse(self.probabilities, global_w)[0], [17 / 30, 13 / 30])
        assert_allclose(fuse(self.probabilities, global_w), fuse(self.probabilities, local_w[[0, 0, 1, 1]]))

    def test_distinct_local_weights_and_actual_zero_region(self):
        probabilities = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [0., 1.]]])
        _, local_c, global_w, local_w = estimate_competence(probabilities, np.array([0, 0]), np.array([0, 1]), 2)
        assert_allclose(local_c, [[1, 0], [0, 0]])
        assert_allclose(local_w[0], [1, 0])
        assert_allclose(local_w[1], global_w)

    def test_probability_alignment_and_ties(self):
        class ReverseModel:
            classes_ = np.array([1, 0])

            def predict_proba(self, _):
                return np.array([[0.1, 0.9], [0.5, 0.5]])

        aligned = aligned_probabilities(ReverseModel(), None)
        assert_allclose(aligned, [[0.9, 0.1], [0.5, 0.5]])
        assert_array_equal(aligned.argmax(axis=1), [0, 0])


class ExcessCompetenceTests(unittest.TestCase):
    def test_model_at_majority_rate_gets_no_weight(self):
        # 8 negatives, 2 positives: majority rate 0.8.
        labels = np.array([0] * 8 + [1] * 2)
        correct = np.column_stack([np.ones(10, bool), np.r_[np.ones(8, bool), np.zeros(2, bool)],
                                   np.r_[np.ones(9, bool), np.zeros(1, bool)]])
        assert_allclose(global_excess_weights(correct, labels), [2 / 3, 0, 1 / 3])

    def test_no_model_above_majority_uses_fallback(self):
        weights = excess_weights([[5, 4]], [10], [5], [0.5, 0.4], 0.5, 0, np.array([0.3, 0.7]))
        assert_allclose(weights, [[0.3, 0.7]])

    def test_shrinkage_limits(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        correct = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [1, 1], [1, 1]], dtype=bool)
        regions = np.array([0, 0, 0, 1, 1, 1])
        statistics = cluster_statistics(correct, labels, regions, np.array([0, 1]), 2)
        assert_allclose(local_excess_weights(correct, labels, *statistics, 1e9),
                        np.tile(global_excess_weights(correct, labels), (2, 1)), atol=1e-6)
        # Without shrinkage, region 0 (rates 1.0 and 1/3, majority 2/3) gives all weight to model 0.
        assert_allclose(local_excess_weights(correct, labels, *statistics, 0)[0], [1, 0])

    def test_single_cluster_reproduces_global_weights(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 60)
        correct = rng.random((60, 5)) < 0.8
        statistics = cluster_statistics(correct, labels, np.zeros(60, int), np.zeros(7, int), 1)
        for shrinkage in (0, 10, 30):
            assert_allclose(local_excess_weights(correct, labels, *statistics, shrinkage),
                            np.tile(global_excess_weights(correct, labels), (7, 1)))

    def test_neighbor_statistics(self):
        labels = np.array([0, 1, 1])
        correct = np.array([[1, 0], [1, 1], [0, 1]], dtype=bool)
        hits, counts, positives = neighbor_statistics(correct, labels, np.array([[0, 1], [1, 2]]))
        assert_allclose(hits, [[2, 1], [1, 2]])
        assert_allclose(counts, [2, 2])
        assert_allclose(positives, [1, 2])


class CalibrationTests(unittest.TestCase):
    def test_platt_recovers_monotone_mapping_and_handles_separation(self):
        rng = np.random.default_rng(1)
        scores = rng.random(400)
        labels = (rng.random(400) < scores ** 3).astype(int)
        params = fit_platt(scores, labels)
        calibrated = apply_platt(params, np.sort(scores))[:, 1]
        self.assertTrue(np.all(np.diff(calibrated) >= 0))
        separable = fit_platt(np.r_[np.zeros(20), np.ones(20)], np.r_[np.zeros(20, int), np.ones(20, int)])
        self.assertTrue(np.isfinite(separable).all())

    def test_crossfit_calibration_ignores_own_fold_labels(self):
        rng = np.random.default_rng(2)
        oof = rng.random((90, 2))
        oof = np.stack([1 - oof, oof], axis=2)
        labels = rng.integers(0, 2, 90)
        assignment = np.repeat([1, 2, 3], 30)
        first, _, _ = calibrate(oof, labels, assignment, oof[:5])
        changed = labels.copy()
        changed[assignment == 1] = 1 - changed[assignment == 1]
        second, _, _ = calibrate(oof, changed, assignment, oof[:5])
        assert_allclose(first[assignment == 1], second[assignment == 1])
        self.assertFalse(np.allclose(first[assignment == 2], second[assignment == 2]))


class RegionSelectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.points = rng.normal(size=(120, 3))
        self.labels = (self.points[:, 0] + rng.normal(scale=0.5, size=120) > 0).astype(int)
        positive = 1 / (1 + np.exp(-rng.normal(size=(120, 5)) - 2 * (self.labels[:, None] - 0.5)))
        self.oof = np.stack([1 - positive, positive], axis=2)
        self.assignment = np.tile([1, 2, 3], 40)
        self.test_points = rng.normal(size=(15, 3))
        test_positive = rng.random((15, 5))
        self.test = np.stack([1 - test_positive, test_positive], axis=2)

    def test_cluster_selection_and_k1_equivalence(self):
        regions = ClusterRegions(self.points, seed=0)
        scores, weights, (k, shrinkage), assignment, selection = select_and_fuse(
            regions, self.oof, self.labels, self.assignment, self.test, self.test_points)
        self.assertEqual(len(selection), len(ClusterRegions.candidates))
        self.assertEqual(sum(r["selected"] for r in selection), 1)
        assert_allclose(weights.sum(axis=1), 1)
        self.assertTrue((weights >= 0).all())
        self.assertTrue(set(assignment) <= set(range(k)))
        correct = self.oof.argmax(axis=2) == self.labels[:, None]
        single = local_excess_weights(correct, self.labels, *cluster_statistics(
            correct, self.labels, regions.labels[1], np.zeros(15, int), 1), 30)
        assert_allclose(fuse(self.test, single), fuse(self.test, global_excess_weights(correct, self.labels)))

    def test_neighbor_prefix_and_containment(self):
        regions = NeighborRegions(self.points)
        rest, held = np.flatnonzero(self.assignment != 1), np.flatnonzero(self.assignment == 1)
        context = regions.prepare(rest, held)
        self.assertLess(context.max(), len(rest))
        _, _, (k, _), neighbors, _ = select_and_fuse(regions, self.oof, self.labels, self.assignment,
                                                     self.test, self.test_points)
        self.assertEqual(neighbors.shape, (15, k))
        self.assertLess(neighbors.max(), len(self.points))


class ProtocolTests(unittest.TestCase):
    def test_nested_groups_and_coverage(self):
        groups = np.repeat(np.arange(60), 2)
        labels = np.repeat(np.arange(60) % 2, 2)
        coverage = np.zeros(120)
        for train, test in make_splits(labels, groups, 5, seed=7):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            coverage[test] += 1
            inner_coverage = np.zeros(len(train))
            for inner_train, validation in make_splits(labels[train], groups[train], 3, seed=7):
                self.assertFalse(set(groups[train[inner_train]]) & set(groups[train[validation]]))
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
        self.assertEqual(values["balanced_accuracy"], .5)
        self.assertTrue(np.isnan(values["precision"]))
        self.assertIn("precision", values["undefined"])

    def test_diversity_measures(self):
        labels = np.array([0, 1, 0, 1])
        probabilities = np.zeros((4, 2, 2))
        probabilities[:, 0] = [[1, 0], [0, 1], [1, 0], [1, 0]]
        probabilities[:, 1] = [[0, 1], [0, 1], [1, 0], [0, 1]]
        values = diversity(probabilities, labels)
        self.assertEqual(values["oracle_accuracy"], 1)
        self.assertEqual(values["mean_disagreement"], 0.5)

    def test_all_models_and_routers_on_real_data(self):
        from src.datasets import load_dataset
        from src.hybrid import RevisedRouter
        features, labels, groups, _ = load_dataset("sonar")
        train, test = make_splits(labels, groups, 5, seed=11)[0]
        router = CompetenceRouter(seed=11).fit(features.iloc[train])
        self.assertEqual(len(router.predict(features.iloc[test])), len(test))
        diagnostics = router.diagnostics()
        self.assertEqual(sum(r["size"] for r in diagnostics), len(train))
        total_ss = ((router.training_features_ - router.training_features_.mean(axis=0)) ** 2).sum()
        assert_allclose(sum(r["within_ss"] + r["between_ss"] for r in diagnostics), total_ss)
        revised = RevisedRouter(seed=11).fit(features.iloc[train])
        self.assertEqual(len(revised.diagnostics()), sum(CLUSTER_CANDIDATES))
        self.assertTrue(np.isfinite(revised.transform(features.iloc[test])).all())
        models = make_models(seed=11)
        self.assertEqual(list(models), BASE_MODELS)
        for model in models.values():
            model.fit(features.iloc[train], labels[train])
            self.assertEqual(aligned_probabilities(model, features.iloc[test]).shape, (len(test), 2))


class StatisticsTests(unittest.TestCase):
    def test_incomplete_statistics_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                compare(pd.DataFrame([{"dataset": "sonar", "model": "M1", "repetition": 1, "fold": 1,
                                       "roc_auc": .7}]), Path(directory))

    def test_holm_and_paired_test(self):
        assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        result = paired_test(np.array([.9, .8, .7, .6]), np.array([.8, .8, .6, .5]))
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (3, 0, 1))
        self.assertEqual(paired_test(np.ones(5), np.ones(5))["wilcoxon_p"], 1.0)

    def test_critical_difference_matches_table(self):
        # Demšar (2006), Table 5: q_0.05 = 2.949 for 7 classifiers and 3.031 for 8.
        self.assertAlmostEqual(critical_difference(7, 10)[0], 2.949, delta=1e-3)
        self.assertAlmostEqual(critical_difference(8, 10)[0], 3.031, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
