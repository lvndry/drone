"""BP metadata allowlist: IDs/genres/subgenres/duration only, never key fields."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import resource
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED = {
    "workbook": "756ec8c9709a50c03c1c741cf051e2b0bb984df3dc914ec733933cd92cd584dc",
    "store_json_zip": "cd3d7c144dec0031581531315ebc7e340e8ac89161254f37a4167bea77ad13c3",
    "plan": "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373",
    "canonical_counts": "7d56908dc5e84c165c7844d1e810ed678123ae7d0a93939c60a2e600b6402b4c",
}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def band(seconds: float) -> str:
    return (
        "<30s"
        if seconds < 30
        else "30s–<2m"
        if seconds < 120
        else "2–<5m"
        if seconds < 300
        else "5–10m"
        if seconds <= 600
        else ">10m"
    )


def selected_cell(cell: ET.Element | None, column: str) -> tuple[str, str] | None:
    if column not in ("A", "F"):
        raise ValueError("Only BP id/genre columns A/F are admitted")
    if cell is None:
        return None
    if not re.fullmatch(column + r"\d+", cell.get("r", "")):
        raise ValueError("Wrong spreadsheet column")
    assert cell.find("m:f", NS) is None, "selected formula rejected"
    if cell.get("t") == "inlineStr":
        inline = cell.find("m:is", NS)
        assert inline is not None
        return "text", "".join(inline.itertext())
    value = cell.find("m:v", NS)
    return None if value is None else (cell.get("t", "n"), value.text or "")


def resolve_shared(raw: bytes, wanted: set[int]) -> dict[int, str]:
    tree = ET.fromstring(raw)
    return {index: "".join(item.itertext()) for index, item in enumerate(tree) if index in wanted}


def resolved(cell: tuple[str, str] | None, shared: dict[int, str]) -> str:
    if cell is None:
        return ""
    return shared[int(cell[1])] if cell[0] == "s" else cell[1]


def parse_tags(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    assert isinstance(value, list)
    result = []
    for tag in value:
        assert isinstance(tag, dict)
        # Nested native taxonomy descriptors only; preserve IDs/slugs/names verbatim.
        chosen = {key: tag.get(key) for key in ("id", "name", "slug")}
        assert isinstance(chosen["name"], str)
        result.append(chosen)
    return result


def selected_store_fields(raw: bytes) -> tuple[dict[str, Any], bool]:
    """Return only id/genres/subgenres/duration; never return store/reference key fields."""
    text = raw.decode()
    normalized = False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        token = re.compile(r'"(?:[^"\\]|\\.)*"|(?<![\w])nan(?![\w])')
        changed = token.sub(lambda found: "null" if found[0] == "nan" else found[0], text)
        if changed == text:
            raise
        payload = json.loads(changed)
        normalized = True
    return {key: payload.get(key) for key in ("id", "genres", "sub_genres", "duration")}, normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.inputs.read_bytes())["bp"]
    paths = {name: Path(row["path"]) for name, row in records.items()}
    for name, path in paths.items():
        if sha(path.read_bytes()) != records[name]["sha256"]:
            raise ValueError("Existing BP metadata source changed: " + name)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    raws = {name: path.read_bytes() for name, path in paths.items()}
    sources = {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in raws.items()}
    for name, expected in EXPECTED.items():
        assert sources[name]["sha256"] == expected
    text = gzip.decompress(raws["plan"]).decode()
    roles = {}
    for field, role in [("bp_fit", "fit"), ("bp_selection_guard", "inner_selection")]:
        match = re.search(r'"' + field + r'"\s*:\s*(\[[^\]]*\])', text)
        assert match is not None
        roles[role] = set(json.loads(match[1]))
    assert len(roles["fit"]) == 1229 and len(roles["inner_selection"]) == 126
    wanted = roles["fit"] | roles["inner_selection"]
    assert len(wanted) == 1355
    groups = {
        row["id"]: row["derived_uncertainty_group"]
        for row in csv.DictReader(io.StringIO(gzip.decompress(raws["groups"]).decode()))
        if row["domain"] == "beatport"
    }
    assert set(groups) == wanted
    manifest = json.loads(raws["canonical_counts"])
    # Do not select any target/reference fields. Existing ID/sample-count metadata only.
    samples = {row["id"]: row["samples"] for row in manifest["rows"] if row["id"] in wanted}
    assert set(samples) == wanted
    with zipfile.ZipFile(io.BytesIO(raws["workbook"])) as book:
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
        shared_raw = book.read("xl/sharedStrings.xml")
        id_cells = []
        for row in sheet.findall("m:sheetData/m:row", NS):
            cell = next(
                (candidate for candidate in row if re.fullmatch(r"A\d+", candidate.get("r", ""))),
                None,
            )
            id_cells.append((row, selected_cell(cell, "A")))
        shared_ids = resolve_shared(
            shared_raw,
            {
                int(selected_id[1])
                for _, selected_id in id_cells
                if selected_id is not None and selected_id[0] == "s"
            },
        )
        selected = []
        for row, cell in id_cells:
            identity = resolved(cell, shared_ids)
            if not identity.isdigit() or "beatport:" + str(int(identity)) not in wanted:
                continue
            native = next(
                (candidate for candidate in row if re.fullmatch(r"F\d+", candidate.get("r", ""))),
                None,
            )
            selected.append(("beatport:" + str(int(identity)), selected_cell(native, "F")))
        shared_genres = resolve_shared(
            shared_raw,
            {
                int(genre_cell[1])
                for _, genre_cell in selected
                if genre_cell is not None and genre_cell[0] == "s"
            },
        )
        sheet_genres = {
            identity: resolved(cell, shared_genres).strip() for identity, cell in selected
        }
        assert len(sheet_genres) == len(selected)
    api: dict[str, dict[str, Any]] = {}
    normalized = []
    with zipfile.ZipFile(io.BytesIO(raws["store_json_zip"])) as bundle:
        for member in bundle.namelist():
            match = re.fullmatch(r"meta/(\d+) .+\.json", member)
            if match is None or "beatport:" + match[1] not in wanted:
                continue
            raw = bundle.read(member)
            chosen, changed = selected_store_fields(raw)
            if changed:
                normalized.append("beatport:" + match[1])
            identity = "beatport:" + str(chosen["id"])
            assert identity == "beatport:" + match[1] and identity not in api
            duration = chosen["duration"]
            milliseconds = None
            minute_text = None
            if duration is not None:
                assert isinstance(duration, dict)
                milliseconds = duration.get("milliseconds")
                minute_text = duration.get("minutes")
                assert milliseconds is None or (type(milliseconds) is int and milliseconds >= 0)
            api[identity] = {
                "genres": parse_tags(chosen["genres"]),
                "sub_genres": parse_tags(chosen["sub_genres"]),
                "duration_milliseconds": milliseconds,
                "duration_minutes_text": minute_text,
                "json_member_sha256": sha(raw),
            }

    def stats(ids: set[str]) -> dict[str, Any]:
        return {
            "tracks": len(ids),
            "declared_uncertainty_groups": len({groups[identity] for identity in ids}),
            "record_unknown_artist_proxy_tracks": sum(
                groups[identity].startswith("bp-record-unknown-artist:") for identity in ids
            ),
            "artist_album_components_verified": False,
            "preview_bands": dict(
                sorted(Counter(band(samples[identity] / 24000) for identity in ids).items())
            ),
        }

    summaries = {}
    workbook_inventory = {}
    api_inventory = {}
    sub_inventory = {}
    for role, ids in roles.items():
        present = ids & api.keys()
        tagged = {identity for identity in ids if sheet_genres.get(identity)}
        summaries[role] = {
            **stats(ids),
            "workbook_rows": len(ids & sheet_genres.keys()),
            "workbook_genres_nonempty": len(tagged),
            "api_json_present": len(present),
            "api_json_missing": len(ids - present),
            "api_genres_nonempty": sum(bool(api[identity]["genres"]) for identity in present),
            "api_subgenres_nonempty": sum(
                bool(api[identity]["sub_genres"]) for identity in present
            ),
            "workbook_not_equal_to_any_api_genre": sum(
                bool(sheet_genres.get(identity))
                and sheet_genres[identity]
                not in {genre["name"] for genre in api[identity]["genres"]}
                for identity in present
            ),
            "actual_preview_seconds_min": min(samples[identity] / 24000 for identity in ids),
            "actual_preview_seconds_max": max(samples[identity] / 24000 for identity in ids),
            "store_duration_known": sum(
                api[identity]["duration_milliseconds"] is not None for identity in present
            ),
            "store_duration_bands": dict(
                sorted(
                    Counter(
                        band(api[identity]["duration_milliseconds"] / 1000)
                        for identity in present
                        if api[identity]["duration_milliseconds"] is not None
                    ).items()
                )
            ),
        }
    for name in sorted({genre for genre in sheet_genres.values() if genre}):
        members = {identity for identity, genre in sheet_genres.items() if genre == name}
        workbook_inventory[name] = {role: stats(ids & members) for role, ids in roles.items()}
    for field, output in [("genres", api_inventory), ("sub_genres", sub_inventory)]:
        for name in sorted({tag["name"] for row in api.values() for tag in row[field]}):
            members = {
                identity
                for identity, row in api.items()
                if name in {tag["name"] for tag in row[field]}
            }
            output[name] = {role: stats(ids & members) for role, ids in roles.items()}
    duration_mismatch = [
        identity
        for identity, row in api.items()
        if row["duration_milliseconds"] is not None
        and abs(row["duration_milliseconds"] / 1000 - samples[identity] / 24000) > 1
    ]
    minute_inconsistent = []
    for identity, row in api.items():
        text = row["duration_minutes_text"]
        milliseconds = row["duration_milliseconds"]
        if text and milliseconds is not None:
            clock = re.fullmatch(r"(\d+):(\d{2})", text)
            if (
                clock is None
                or abs((int(clock[1]) * 60 + int(clock[2])) - milliseconds / 1000) >= 1
            ):
                minute_inconsistent.append(identity)
    result = {
        "kind": "BP-native-genre-existing-metadata-audit-v1",
        "sources": sources,
        "roles": summaries,
        "selection_policy": "Current c3e1229fit+126selection only; original1230 totals are not reused.",
        "field_allowlist": "Workbook A:id and F:genres ONLY; resolve shared strings only for selected A/Fcells; selected formulas rejected. JSON id,genres,sub_genres,duration only, after membership filter. No key/reference/confidence/artist/title field selected.",
        "group_policy": "Current frozen uncertainty groups only; record-unknown-artist singleton proxies are NOT verified artist/album components.",
        "genre_policy": "Preserve native workbook exact strings and API genre/subgenre descriptors separately. No FMA taxonomy conversion or inferred parent labels.",
        "duration_policy": "Prepared waveform sample-count/24000 is actual preview duration. API explicitly supplies milliseconds and minutes text for store recording; never substitute it for evaluated preview length.",
        "workbook_genres": workbook_inventory,
        "api_genres": api_inventory,
        "api_subgenres": sub_inventory,
        "lowercase_bare_nan_normalized_count": len(normalized),
        "store_duration_differs_from_actual_preview_more_than1s": len(duration_mismatch),
        "store_minutes_text_inconsistent_with_milliseconds_by1s_or_more": len(minute_inconsistent),
        "key_labels_selected_or_scored": False,
        "audio_models_or_downloads_used": False,
        "pool_changes": False,
    }
    output = output_dir / "coverage-bp.json"
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    use = resource.getrusage(resource.RUSAGE_SELF)
    receipt = {
        "script_sha256": sha(Path(__file__).read_bytes()),
        "result_sha256": sha(output.read_bytes()),
        "result_bytes": output.stat().st_size,
        "wall_seconds": time.monotonic() - started,
        "self_cpu_seconds": use.ru_utime + use.ru_stime,
        "peak_rss_bytes": use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024,
    }
    (output_dir / "execution-bp.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": receipt,
                "roles": summaries,
                "workbook_genres": workbook_inventory,
                "api_subgenres": sub_inventory,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
