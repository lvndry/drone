"""Metadata lineage and nonexclusive genre semantics for the fixed native panel."""

from __future__ import annotations

import gzip
import json

import pytest

from experiments.collection_catalog import CollectionCatalog, load_collection
from tools.describe_native_panel import CATALOGS, ROOT, STUDY, analyze, band, join


@pytest.fixture(scope="module")
def catalogs() -> dict[str, CollectionCatalog]:
    return {
        source: load_collection(
            ROOT / "experiments/studies/v2-data-contract/data" / f"{stem}-collection.json.gz"
        )
        for source, (stem, _digest, _count) in CATALOGS.items()
    }


def test_existing_metadata_reproduces_published_description() -> None:
    observed = analyze()
    stored = json.loads(gzip.decompress((ROOT / STUDY / "public376-metadata.json.gz").read_bytes()))
    # JSON encoding also normalizes dataclass tuples to their stored array representation.
    assert json.loads(json.dumps(observed)) == stored
    assert (
        observed["audio_read"] is observed["model_executed"] is observed["key_labels_read"] is False
    )
    fma = observed["sources"]["fmak"]
    assert fma["native_top_missing"] == 125
    assert fma["any_genre_tags_known"] == 249
    tags = {
        row["name"]: row["tracks"]
        for row in fma["tags_by_duration_nonexclusive"]["5_to_under_10_min"]
    }
    assert (tags["Electronic"], tags["Rock"], tags["Pop"]) == (22, 18, 10)
    assert sum(tags[name] for name in ["Electronic", "Rock", "Pop"]) > 41


def test_same_id_with_different_audio_is_rejected(catalogs: dict[str, CollectionCatalog]) -> None:
    first = json.loads((ROOT / STUDY / "FUSED_376_MEMBERSHIP.json").read_bytes())["cases"][0]
    with pytest.raises(ValueError, match="hash mismatch"):
        join([first | {"audio_sha256": "f" * 64}], catalogs)


def test_duplicate_panel_entry_is_rejected(catalogs: dict[str, CollectionCatalog]) -> None:
    first = json.loads((ROOT / STUDY / "FUSED_376_MEMBERSHIP.json").read_bytes())["cases"][0]
    with pytest.raises(ValueError, match="duplicate"):
        join([first, first | {"index": 1}], catalogs)


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (119.999, "under_2_min"),
        (120, "2_to_under_5_min"),
        (300, "5_to_under_10_min"),
        (600, "10_min_or_longer"),
    ],
)
def test_native_duration_boundaries(seconds: float, expected: str) -> None:
    assert band(seconds) == expected
