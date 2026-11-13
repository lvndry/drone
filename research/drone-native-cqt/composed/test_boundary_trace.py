"""Preserve the diagnostic counterexample and public metadata integrity."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from fused376_transport import module_path
from summarize_boundary_trace import PINS, checked, summarize
from summarize_fused_product import summarize as summarize_candidate
from summarize_fused_product import summarize_goldens, summarize_public

ROOT = Path(__file__).resolve().parent


class BoundaryTraceTests(unittest.TestCase):
    def test_product_rounding_explains_responses_without_promoting_core(self) -> None:
        raw = (ROOT / "csr-product-rounding-results.json").read_bytes()
        provenance = json.loads((ROOT / "csr-product-rounding-provenance.json").read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), provenance["results_sha256"])
        result = json.loads(raw)
        prior = summarize(ROOT)["complete_array_comparisons"]
        self.assertEqual([stage["stage"] for stage in result["stages"]], list(range(9)))
        for stage in result["stages"]:
            self.assertTrue(stage["existing_scipy_response_exact"])
            self.assertEqual(stage["used_spectrum_unequal_complex"], 0)
            self.assertEqual(stage["strict_native_unequal"], 0)
            self.assertEqual(stage["fused_python_unequal"], 0)
            self.assertEqual(
                stage["opposite_unequal"], prior[f"stage{stage['stage']}.response"]["unequal_bits"]
            )
        self.assertFalse(result["new_final_core_computed"])
        self.assertFalse(result["candidate_compiled_or_run"])
        self.assertEqual(prior["terminal.fp16"]["unequal_bits"], 9)

    def test_compiled_candidate_preserves_the_negative_control(self) -> None:
        result = summarize_candidate(ROOT)
        self.assertEqual(result["original_differences"], 9)
        self.assertEqual(result["candidate_differences"], 0)
        self.assertFalse(result["new_accuracy_or_portability_claim"])

    def test_candidate_claim_requires_the_original_result_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("fused-product-results.json", "fused-product-provenance.json"):
                shutil.copyfile(ROOT / name, root / name)
            path = root / "fused-product-results.json"
            result = json.loads(path.read_bytes())
            result["baseline"]["python_comparison"]["unequal_bits"] = 0
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "identity changed"):
                summarize_candidate(root)

    def test_golden_candidate_preserves_all_original_controls(self) -> None:
        result = summarize_goldens(ROOT)
        self.assertEqual(len(result["cases"]), 9)
        self.assertEqual(result["core_elements"], 2048256)
        self.assertEqual(result["original_differences"], 303)
        self.assertEqual(result["candidate_differences"], 0)
        self.assertGreater(result["cases"][-1]["decoded_seconds"], 1682)
        self.assertFalse(result["new_accuracy_or_portability_claim"])

    def test_golden_membership_cannot_drop_the_long_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("fused-goldens-results.json", "fused-goldens-provenance.json"):
                shutil.copyfile(ROOT / name, root / name)
            path = root / "fused-goldens-results.json"
            result = json.loads(path.read_bytes())
            result["cases"].pop()
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "identity changed"):
                summarize_goldens(root)

    def test_archived_support_requires_a_file_origin(self) -> None:
        module = ModuleType("missing_origin")
        module.__file__ = None
        with self.assertRaisesRegex(ValueError, "no file origin"):
            module_path(module)
        module.__file__ = "archived_support.py"
        self.assertEqual(module_path(module), Path("archived_support.py"))

    def test_public_panel_keeps_both_references_and_long_inputs(self) -> None:
        result = summarize_public(ROOT)
        self.assertEqual(result["totals"]["candidate_mac_exact"], 376)
        self.assertEqual(result["totals"]["controls_preserved"], 376)
        self.assertEqual(result["totals"]["candidate_linux_exact_secondary"], 0)
        self.assertEqual(result["totals"]["core_elements"], 82562688)
        self.assertEqual(result["groups"]["10_min_or_longer"]["cases"], 9)
        self.assertFalse(result["new_accuracy_or_portability_claim"])

    def test_public_panel_cannot_drop_a_mismatching_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "fused-376-results.json"
            result = json.loads((ROOT / path.name).read_bytes())
            result["cases"][0]["candidate_linux_exact_secondary"] = True
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "identity changed"):
                summarize_public(root)

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
        for name in [
            *PINS,
            "csr-product-rounding-results.json",
            "csr-product-rounding-provenance.json",
            "fused-product-results.json",
            "fused-product-provenance.json",
            "fused-goldens-results.json",
            "fused-goldens-provenance.json",
            "fused-376-results.json",
            "fused-376-provenance.json",
            "fused-376-controller.json",
            "fused-376-server-controller.json",
            "FUSED_376_MEMBERSHIP.json",
        ]:
            raw = (ROOT / name).read_text()
            for private in ("/Users/", "/home/", "/private/", "/tmp/", '"pid"', '"start_abstime"'):
                self.assertNotIn(private, raw)
        result = summarize(ROOT)
        for row in result["different_core_elements"]:
            self.assertNotIn("core", row)
            self.assertNotIn("magnitude", row)


if __name__ == "__main__":
    unittest.main()
