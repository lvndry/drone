"""Pick the drone-v2 seed and decide on shift averaging using development data only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontier-distillation" / "scripts"))
import numpy as np  # noqa: E402
from analyze import load_panels  # noqa: E402
from compare import SHIFTS, interval, is_test_half  # noqa: E402

SEEDS = [13, 15, 42]


def correctness(stack: np.ndarray, references: list[int], shift_average: bool) -> np.ndarray:
    logits = stack.mean(1) if shift_average else stack[:, SHIFTS.index(0)]
    return (logits.argmax(-1) == np.array(references)).astype(float)


def main() -> None:
    panels = load_panels()
    stored = np.load("analysis-r6.npz")
    generator = np.random.default_rng(0)
    for seed_index, seed in enumerate(SEEDS):
        single_parts = []
        averaged_parts = []
        test_single = []
        test_averaged = []
        for panel in ("valid", "holdout", "gsplus_primary", "gsplus_clean"):
            _, references, track_ids = panels[panel]
            stack = stored[f"shift12/{panel}"][seed_index]
            single = correctness(stack, references, False)
            averaged = correctness(stack, references, True)
            if panel.startswith("gsplus"):
                test = np.array([is_test_half(track) for track in track_ids])
                if panel == "gsplus_primary":
                    single_parts.append(single[~test])
                    averaged_parts.append(averaged[~test])
                else:
                    test_single.append(single[test].mean())
                    test_averaged.append(averaged[test].mean())
            else:
                single_parts.append(single)
                averaged_parts.append(averaged)
        single_all = np.concatenate(single_parts)
        averaged_all = np.concatenate(averaged_parts)
        low, high = interval(averaged_all - single_all, generator)
        print(
            f"seed {seed}: development single {100 * single_all.mean():.2f} "
            f"shift-averaged {100 * averaged_all.mean():.2f} "
            f"difference {100 * (averaged_all - single_all).mean():+.2f} [{100 * low:+.2f}, {100 * high:+.2f}] "
            f"| clean test single {100 * test_single[0]:.2f} averaged {100 * test_averaged[0]:.2f}"
        )


if __name__ == "__main__":
    main()
