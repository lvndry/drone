import json
import sys
from pathlib import Path

groups = sys.argv[1:] or ["control", "sslinit", "consistency", "distill", "wide2", "longcrop"]
for group in groups:
    for seed in (13, 15, 42):
        path = Path("runs") / f"{group}-s{seed}" / "result.json"
        if not path.is_file():
            print(f"{group:12s} {seed} MISSING")
            continue
        result = json.loads(path.read_text())
        row = " ".join(f"{panel}={values['exact']}" for panel, values in result["selected"].items())
        print(f"{group:12s} {seed} ep{result['best_epoch']:3d} {row}")
