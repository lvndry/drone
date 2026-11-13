"""Preserve the diagnostic counterexample and public metadata integrity."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from summarize_boundary_trace import PINS, checked, summarize

ROOT = Path(__file__).resolve().parent


class BoundaryTraceTests(unittest.TestCase):
    def test_archived_failure_is_localized_not_promoted(self) -> None:
        result = summarize(ROOT)
        arrays = result["complete_array_comparisons"]
        self.assertEqual(len(arrays), 41)
        self.assertEqual(len(result["probe_comparisons"]), 48)
        self.assertEqual(arrays["stage0.spectrum"]["unequal_bits"], 0)
        self.assertEqual(arrays["stage0.response"]["unequal_bits"], 6438)
        self.assertEqual(arrays["terminal.fp16"]["unequal_bits"], 9)
        self.assertEqual(result["stage5_fft_difference_csr_support_rows"], [])
        self.assertTrue(
            all(row["normalization_scale_equal_bits"] for row in result["different_core_elements"])
        )
        self.assertTrue(result["resource_receipt"]["no_model_or_scoring"])
        self.assertEqual(len(result["historical_gaps"]), 4)

    def test_changed_diagnostic_cannot_relabel_the_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in PINS:
                shutil.copyfile(ROOT / name, root / name)
            path = root / "boundary-trace-results.json"
            result = json.loads(path.read_bytes())
            result["complete_array_comparisons"]["terminal.fp16"]["unequal_bits"] = 0
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "identity changed"):
                summarize(root)

    def test_oversized_artifact_is_refused_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large"
            path.write_bytes(b"12345")
            with self.assertRaisesRegex(ValueError, "size bound"):
                checked(path, "unused", maximum=4)

    def test_public_metadata_contains_no_operational_paths_or_features(self) -> None:
        for name in PINS:
            raw = (ROOT / name).read_text()
            for private in ("/Users/", "/home/", "/private/", "/tmp/", '"pid"', '"start_abstime"'):
                self.assertNotIn(private, raw)
        result = summarize(ROOT)
        for row in result["different_core_elements"]:
            self.assertNotIn("core", row)
            self.assertNotIn("magnitude", row)


if __name__ == "__main__":
    unittest.main()
