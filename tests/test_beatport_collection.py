"""Label-free BP adapter provenance and allowlist regressions."""

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tools.export_beatport_collection import observed_entities, store_metadata


def bundle(payload: dict[str, Any], identity: str = "123") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(f"meta/{identity} Example.json", json.dumps(payload))
    return output.getvalue()


def test_store_allowlist_preserves_native_entities_without_targets() -> None:
    raw = bundle(
        {
            "id": 123,
            "key": "SECRET_TARGET",
            "bpm": 999,
            "genres": [{"id": 4, "name": "Native", "slug": "native"}],
            "sub_genres": [],
            "artists": [{"id": 8, "name": "Artist"}],
            "remixers": [],
            "release": {"id": 9, "name": "Release"},
            "duration": {"milliseconds": 331000},
        }
    )
    row = store_metadata(raw, {"beatport:123"})["beatport:123"]
    assert row["artists"] == [{"id": 8, "name": "Artist", "slug": None}]
    assert row["release"][0]["id"] == 9
    assert row["catalog_duration_seconds"] == 331
    assert row["genres"][0]["name"] == "Native"
    assert "key" not in row and "bpm" not in row
    assert "SECRET_TARGET" not in json.dumps(row)


def test_store_checks_member_identity_and_membership() -> None:
    assert store_metadata(bundle({"id": 123}), {"beatport:456"}) == {}
    with pytest.raises(ValueError, match="identity mismatch"):
        store_metadata(bundle({"id": 456}), {"beatport:123"})


def test_missing_native_metadata_is_explicit_and_bad_duration_rejected() -> None:
    row = store_metadata(bundle({"id": 123}), {"beatport:123"})["beatport:123"]
    assert row["catalog_duration_seconds"] is None
    assert row["artists"] == row["remixers"] == row["release"] == []
    with pytest.raises(ValueError, match="Invalid catalog duration"):
        store_metadata(bundle({"id": 123, "duration": {"milliseconds": True}}), {"beatport:123"})
    with pytest.raises(ValueError, match="entity list"):
        observed_entities("artist name is not a grouping")


def test_published_collection_keeps_coverage_and_exposure_semantics() -> None:
    from experiments.collection_catalog import load_collection
    from tools.export_beatport_collection import OUTPUT, ROOT, coverage, sha

    catalog = load_collection(ROOT / OUTPUT)
    summary = coverage(catalog)
    assert summary["tracks"] == 1355
    assert summary["roles"] == {"fit": 1229, "inner_selection": 126}
    assert (summary["bpm_known"], summary["bpm_unknown"]) == (716, 639)
    assert summary["catalog_duration_known"] == 676
    assert summary["source_duration_known"] == 0
    assert summary["frozen_groups"] == 1355
    assert all(
        row.analyzed_audio.support_kind == "recorded24khztrainingcache" for row in catalog.tracks
    )
    assert all(row.analyzed_audio.sample_rate == 24000 for row in catalog.tracks)
    assert all(len(row.exposures) == 2 for row in catalog.tracks)
    assert all(
        row.exposures[1].relation == "source_hash_in_release_audio_list_role_not_inferred"
        for row in catalog.tracks
    )
    for source in catalog.sources:
        assert sha((ROOT / source.path).read_bytes()) == source.sha256
    assert {source.source_id for source in catalog.external_sources} == {
        "workbook",
        "store_json_zip",
        "canonical_counts",
    }
    from dataclasses import asdict

    def reject_targets(value: object) -> None:
        if isinstance(value, dict):
            assert not ({"key", "key_label", "target", "prediction", "class_id"} & value.keys())
            for item in value.values():
                reject_targets(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                reject_targets(item)

    reject_targets(asdict(catalog))


def test_external_source_relocation_preserves_pins_and_descriptor_bytes(tmp_path: Path) -> None:
    from pathlib import Path

    from tools.export_beatport_collection import EXTERNAL_IDS, INPUTS, ROOT, pinned_inputs, sha

    records: dict[str, dict[str, Any]] = {}
    replacements = {}
    for name in (*sorted(EXTERNAL_IDS), "plan", "groups"):
        raw = ("pinned metadata " + name).encode()
        original = (
            tmp_path / "old" / name if name in EXTERNAL_IDS else tmp_path / "experiments" / name
        )
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(raw)
        records[name] = {"path": str(original), "sha256": sha(raw), "bytes": len(raw)}
        if name in EXTERNAL_IDS:
            replacement = tmp_path / "relocated" / name
            replacement.parent.mkdir(exist_ok=True)
            replacement.write_bytes(raw)
            replacements[name] = replacement
    descriptor = tmp_path / INPUTS.relative_to(ROOT)
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text(json.dumps({"bp": records}))
    before = pinned_inputs(tmp_path)
    for name in EXTERNAL_IDS:
        Path(records[name]["path"]).unlink()
    after = pinned_inputs(tmp_path, replacements)
    assert before == after  # Payloads and catalog source descriptors remain byte-identical.
    assert all("relocated" not in row["path"] for row in after[1].values())
    with pytest.raises(ValueError, match="exactly"):
        pinned_inputs(tmp_path, {})
    replacements["workbook"].write_bytes(b"different version")
    with pytest.raises(ValueError, match="source mismatch: workbook"):
        pinned_inputs(tmp_path, replacements)


@pytest.mark.parametrize("os_byte", [3, 19, 255])
def test_gzip_wrapper_portability_preserves_archive_and_reports_actual_sha(
    tmp_path: Path, os_byte: int
) -> None:
    import gzip

    from tools.export_beatport_collection import prepare_artifacts, sha

    raw = b'{"stable_metadata": true}\n'
    archived = bytearray(gzip.compress(raw, mtime=0))
    archived[9] = os_byte  # Linux, macOS, portable/unknown gzip OS markers.
    output = tmp_path / "catalog.json.gz"
    output.write_bytes(archived)
    for write in (False, True):
        compressed, summary = prepare_artifacts(raw, {"tracks": 1}, output, write=write)
        assert compressed == bytes(archived) == output.read_bytes()
        assert json.loads(summary)["catalog_sha256"] == sha(bytes(archived))
    with pytest.raises(ValueError, match="canonical payload"):
        prepare_artifacts(b'{"stable_metadata": false}\n', {}, output, write=False)


def test_future_gzip_writes_use_portable_header(tmp_path: Path) -> None:
    import gzip

    from tools.export_beatport_collection import prepare_artifacts, sha

    raw = b'{"new_collection": true}\n'
    compressed, summary = prepare_artifacts(raw, {}, tmp_path / "new.json.gz", write=True)
    assert compressed[9] == 255
    assert gzip.decompress(compressed) == raw
    assert json.loads(summary)["catalog_sha256"] == sha(compressed)


def test_existing_archived_catalog_sha_is_reported_without_recompression() -> None:
    import gzip

    from tools.export_beatport_collection import OUTPUT, ROOT, prepare_artifacts, sha

    output = ROOT / OUTPUT
    archived = output.read_bytes()
    compressed, summary = prepare_artifacts(gzip.decompress(archived), {}, output, write=False)
    assert compressed == archived
    assert sha(archived) == "24a09353dc4e653bfc62898a0ca61ee2ff4838d62c3ea07ee261a2fe4f00b11f"
    assert json.loads(summary)["catalog_sha256"] == sha(archived)
