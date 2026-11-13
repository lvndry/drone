"""Evidence checks and the distinction between numerical fidelity and key agreement."""

import unittest
from pathlib import Path

from summarize_retained_model import comparison, summarize


class RetainedModelEvidenceTests(unittest.TestCase):
    def test_published_evidence_replays_without_inference(self) -> None:
        result = summarize(Path(__file__).resolve().parent)
        self.assertEqual(result["tracks"], 376)
        self.assertTrue(result["passed"])
        self.assertTrue(all(row["strict_passes"] == 376 for row in result["families"].values()))

    def test_same_key_does_not_override_failed_logit_bound(self) -> None:
        expected = [0.0] * 23 + [1.0]
        actual = expected.copy()
        actual[0] = 1.1e-5
        result = comparison(actual, expected, 1e-5, 1e-5)
        self.assertTrue(result["argmax_agrees"])
        self.assertFalse(result["allclose"])
        self.assertEqual(result["failed_elements"], [0])


if __name__ == "__main__":
    unittest.main()
