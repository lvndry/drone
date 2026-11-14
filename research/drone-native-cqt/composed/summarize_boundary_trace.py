"""Verify saved metadata, or replay array comparisons without frontend/model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent
PINS = {
    "boundary-trace-results.json": "d1e5f67a2644703ff22f791cbd988bf05eefc71c7ec39867540c7e6eed45a586",
    "boundary-trace-provenance.json": "d54114b97c0330f5df131b43290d7ee158d7a5b8c0fe7f8fc2a7ed1c5ceea3a9",
}


def checked(path: Path, expected: str, maximum: int = 1_000_000) -> bytes:
    """Refuse changed bytes and oversized inputs before parsing/decompression."""
    if path.stat().st_size > maximum:
        raise ValueError("Artifact exceeds metadata/array size bound")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Artifact identity changed: " + path.name)
    return raw


def summarize(root: Path = ROOT) -> dict[str, Any]:
    data = {name: json.loads(checked(root / name, pin)) for name, pin in PINS.items()}
    result = data["boundary-trace-results.json"]
    if not all(result["archived_core_guards"].values()):
        raise ValueError("Archived cores did not reproduce")
    return result


def compare(python_array: NDArray[Any], native_array: NDArray[Any]) -> dict[str, Any]:
    import numpy as np

    if (
        python_array.shape != native_array.shape
        or python_array.dtype != native_array.dtype
        or not np.isfinite(python_array).all()
    ):
        raise ValueError("Array geometry/dtype/finite contract failed")
    if not np.isfinite(native_array).all():
        raise ValueError("Nonfinite native array")
    equal = same_bits(python_array, native_array)
    difference = np.abs(
        python_array.astype("complex128" if python_array.dtype.kind == "c" else "float64")
        - native_array
    )
    return {
        "shape": list(python_array.shape),
        "dtype": str(python_array.dtype),
        "elements": int(python_array.size),
        "unequal_bits": int((~equal).sum()),
        "unequal_values": int((python_array != native_array).sum()),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs_location": list(
            map(int, np.unravel_index(difference.argmax(), python_array.shape))
        ),
        "python_sha256": hashlib.sha256(python_array.tobytes()).hexdigest(),
        "native_sha256": hashlib.sha256(native_array.tobytes()).hexdigest(),
    }


def same_bits(left: NDArray[Any], right: NDArray[Any]) -> NDArray[np.bool_]:
    if left.dtype.kind == "c":
        return (left.real.view("u4") == right.real.view("u4")) & (
            left.imag.view("u4") == right.imag.view("u4")
        )
    dtype = "u" + str(left.dtype.itemsize)
    return left.view(dtype) == right.view(dtype)


def replay(trace_root: Path, result: dict[str, Any]) -> dict[str, int | str]:
    """Read the original immutable local arrays; never import CQT or ML implementations."""
    import numpy as np

    output = trace_root / "output"
    traces = {
        name: json.loads(checked(output / f"{name}-trace.json", pin))
        for name, pin in result["input_traces"].items()
    }
    cache: dict[str, NDArray[Any]] = {}
    references = 0

    def walk(value: object) -> None:
        nonlocal references
        if isinstance(value, dict):
            if {"path", "sha256", "dtype", "shape", "bytes"} <= value.keys():
                digest = value["sha256"]
                if value["path"] != "objects/" + digest:
                    raise ValueError("Unexpected content-addressed artifact path")
                raw = checked(output / value["path"], digest, maximum=64 * 1024**2)
                if len(raw) != value["bytes"]:
                    raise ValueError("Array byte count differs")
                array = np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite retained array")
                cache[digest] = array
                references += 1
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(traces)
    python = traces["python"]
    native = traces["native"]
    records = dict(python["records"])
    for stage in python["stages"]:
        for source, target in (("input", "input"), ("stft", "spectrum"), ("response", "response")):
            name = f"stage{stage['index']}.{target}"
            records[name] = stage[source]
            for probe_index, probe in enumerate(stage[source].get("probes", [])):
                records[f"{name}.probe{probe_index}"] = probe["artifact"]
    for stage in python["resamplings"]:
        name = f"after_stage{stage['after_stage']}.resampled"
        records[name] = stage["output"]
        for probe_index, probe in enumerate(stage["output"].get("probes", [])):
            records[f"{name}.probe{probe_index}"] = probe["artifact"]

    def array(mapping: dict[str, Any], name: str) -> NDArray[Any]:
        return cache[mapping[name]["sha256"]]

    expected = result["complete_array_comparisons"] | result["probe_comparisons"]
    if set(expected) != set(native["records"]):
        raise ValueError("Comparison coverage changed")
    for name, native_record in native["records"].items():
        actual = compare(array(records, name), cache[native_record["sha256"]])
        for field, value in actual.items():
            target = expected[name][field]
            equal = (
                math.isclose(value, target, rel_tol=1e-12)
                if isinstance(value, float)
                else value == target
            )
            if not equal:
                raise ValueError(f"Comparison changed: {name}.{field}")
    if (
        references != result["blob_records_verified"]
        or len(cache) != result["unique_blobs_verified"]
    ):
        raise ValueError("Artifact coverage changed")
    manifest = json.loads(
        checked(trace_root / "inputs/mac-manifest.json", python["manifest_sha256"])
    )
    case = manifest["cases"][0]
    for observed, bound in zip(python["stages"], case["stages"], strict=True):
        for name in ("basis", "indices", "indptr"):
            item = bound[name]
            path = trace_root / "inputs" / item["path"]
            if not path.resolve().is_relative_to((trace_root / "inputs").resolve()):
                raise ValueError("CSR input escapes retained root")
            checked(path, observed[name]["sha256"])
    lengths = np.array(case["normalization"]["lengths"][8:200], dtype="<f8")
    if lengths.tobytes() != array(records, "terminal.lengths").tobytes():
        raise ValueError("Normalization input lengths differ")
    for mapping in (records, native["records"]):
        stacked = np.concatenate(
            [array(mapping, f"stage{index}.response") for index in range(8, -1, -1)]
        )
        if stacked[8:200].tobytes() != array(mapping, "terminal.pre_normalization").tobytes():
            raise ValueError("Stacking is not the recorded row selection")
    for transition, observation in result["conditional_terminal_comparisons"].items():
        before, after = transition.split(" -> ")
        equal_before = same_bits(
            array(records, "terminal." + before), array(native["records"], "terminal." + before)
        )
        equal_after = same_bits(
            array(records, "terminal." + after), array(native["records"], "terminal." + after)
        )
        actual_counts = {
            "equal_input_elements": int(equal_before.sum()),
            "unequal_output_given_equal_input": int((equal_before & ~equal_after).sum()),
            "unequal_input_equal_output": int((~equal_before & equal_after).sum()),
        }
        if any(observation[key] != count for key, count in actual_counts.items()):
            raise ValueError("Conditional terminal comparison differs")
    differences = np.argwhere(
        ~same_bits(array(records, "terminal.core"), array(native["records"], "terminal.core"))
    )
    if differences.tolist() != [
        [row["bin"], row["frame"]] for row in result["different_core_elements"]
    ]:
        raise ValueError("Final difference coordinates changed")
    stacked_equal = same_bits(
        array(records, "terminal.pre_normalization"),
        array(native["records"], "terminal.pre_normalization"),
    )
    if any(stacked_equal[row["bin"], row["frame"]] for row in result["different_core_elements"]):
        raise ValueError("A final difference was absent at the stacked response")
    scales_equal = same_bits(
        array(records, "terminal.scales"), array(native["records"], "terminal.scales")
    )
    if not all(scales_equal[row["bin"]] for row in result["different_core_elements"]):
        raise ValueError("A final difference has differing normalization scales")
    fft_differences = np.argwhere(
        ~same_bits(array(records, "stage5.spectrum"), array(native["records"], "stage5.spectrum"))
    )
    if fft_differences.tolist() != [[251, 60]]:
        raise ValueError("Unexpected pre-CSR FFT difference")
    indices = cache[python["stages"][5]["indices"]["sha256"]]
    if 251 in indices:
        raise ValueError("The differing FFT frequency is used by CSR")
    return {"status": "passed", "full_arrays": 41, "probes": 48, "verified_references": references}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root", type=Path, help="Optional original local array root; needs NumPy"
    )
    args = parser.parse_args()
    saved = summarize()
    if args.trace_root:
        print(json.dumps(replay(args.trace_root, saved), indent=2))
    else:
        print(
            json.dumps(
                {
                    "case": saved["case"],
                    "finding": saved["interpretation"],
                    "raw_arrays_replayed": False,
                },
                indent=2,
            )
        )
