"""Evaluation reuse must preserve policy identity, suite identity and retained best."""
import copy
import unittest

import numpy as np
import torch

from .evaluation_cache import EvaluationCache, evaluation_key, evaluation_due


class EvaluationCacheTests(unittest.TestCase):
    def setUp(self):
        self.actor = torch.nn.Linear(2, 1)
        self.actor.register_buffer("normalization_mean", torch.zeros(2))
        self.context = dict(backend="native", cases=[dict(speed=1., phase=0.)], episode_seconds=5.)

    def test_copied_inference_state_has_identical_key(self):
        self.assertEqual(evaluation_key(self.actor, self.context),
                         evaluation_key(copy.deepcopy(self.actor), self.context))

    def test_weights_and_normalization_invalidate_cached_result(self):
        original = evaluation_key(self.actor, self.context)
        with torch.no_grad():
            self.actor.weight[0, 0] += .1
        changed = evaluation_key(self.actor, self.context)
        self.assertNotEqual(original, changed)
        self.actor.normalization_mean[0] = 1.
        self.assertNotEqual(changed, evaluation_key(self.actor, self.context))

    def test_suite_backend_and_duration_are_part_of_identity(self):
        original = evaluation_key(self.actor, self.context)
        for changes in (dict(backend="newton"), dict(episode_seconds=30.), dict(cases=[dict(speed=2., phase=0.)])):
            self.assertNotEqual(original, evaluation_key(self.actor, {**self.context, **changes}))

    def test_cache_returns_unmodified_metrics_and_traces(self):
        cache = EvaluationCache()
        result = ({"successes": 8}, {"qpos": np.zeros((2, 3))})
        metrics, traces = cache.get_or_compute("actor", lambda: result)
        metrics["successes"] = 0
        traces["qpos"][:] = 3
        result[0]["successes"] = 1
        again = cache.get_or_compute("actor", lambda: self.fail("Repeated evaluation"))
        self.assertEqual(again[0]["successes"], 8)
        np.testing.assert_array_equal(again[1]["qpos"], 0)
        self.assertEqual(cache.statistics()["hits"], 1)
        self.assertEqual(cache.statistics()["misses"], 1)

    def test_best_survives_eviction_until_replaced(self):
        cache = EvaluationCache(capacity=2)
        cache.get_or_compute("initial", lambda: "best")
        cache.pin("initial")
        for key in ("candidate1", "candidate2", "candidate3"):
            cache.get_or_compute(key, lambda: "candidate")
        self.assertEqual(cache.get_or_compute("initial", lambda: self.fail("Best was evicted")), "best")
        cache.pin("candidate3")
        cache.get_or_compute("candidate4", lambda: "new")
        self.assertEqual(cache.get_or_compute("candidate3", lambda: self.fail("New best was evicted")), "candidate")
        self.assertEqual(cache.statistics()["entries"], 2)

    def test_failed_evaluation_cannot_be_reused(self):
        cache = EvaluationCache()
        def fail():
            raise RuntimeError("Evaluation interrupted")
        with self.assertRaises(RuntimeError):
            cache.get_or_compute("actor", fail)
        self.assertEqual(cache.get_or_compute("actor", lambda: "complete"), "complete")
        self.assertEqual(cache.statistics()["misses"], 1)

    def test_interval_counts_completed_updates_without_update_one_evaluation(self):
        self.assertEqual([i for i in range(130) if evaluation_due(i, 64)], [64, 128])


if __name__ == "__main__":
    unittest.main()
