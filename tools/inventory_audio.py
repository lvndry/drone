"""Inventory a local evaluation library's formats/durations/key tags, without decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

AUDIO = {".mp3", ".m4a", ".flac", ".wav", ".aif", ".aiff", ".ogg", ".opus", ".mp4"}
KEY_TAGS = {"key", "initialkey", "initial_key", "tkey", "----:com.apple.itunes:initialkey"}


def inspect(arguments: tuple[Path, Path]) -> dict[str, Any]:
    root, path = arguments
    row = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        return dict(row, error=result.stderr[:500])
    info = json.loads(result.stdout)
    container = info.get("format", {})
    row["duration_seconds"] = float(container["duration"]) if container.get("duration") else None
    row["format"] = container.get("format_name")
    row["audio_codecs"] = [
        stream.get("codec_name")
        for stream in info.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    tags = {key.lower(): value for key, value in container.get("tags", {}).items()}
    row["key_tags"] = {key: value for key, value in tags.items() if key in KEY_TAGS}
    row["software_tags"] = {
        key: value for key, value in tags.items() if key in {"encoder", "software"}
    }
    with path.open("rb") as stream:
        row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in args.root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO
        and not any(part.startswith(".") for part in path.relative_to(args.root).parts)
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        tracks = list(pool.map(inspect, [(args.root, path) for path in paths]))
    result = {
        "purpose": "local evaluation only; not training data",
        "root": str(args.root),
        "tracks": tracks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            "tracks": len(tracks),
            "formats": dict(Counter(path.suffix.lower() for path in paths)),
            "key_tagged": sum(bool(track.get("key_tags")) for track in tracks),
            "errors": sum("error" in track for track in tracks),
            "unique_audio": len({track["sha256"] for track in tracks if "sha256" in track}),
        }
    )


if __name__ == "__main__":
    main()
