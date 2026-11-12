"""Verify aggregate-only DJ coverage evidence and plot observed genre/duration counts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

STUDY = Path(__file__).parents[1] / "experiments/studies/v2-data-contract"
BANDS = ("<30s", "30s–<2m", "2–<5m", "5–10m", ">10m")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(data: dict[str, Any]) -> dict[str, Any]:
    fma, bp = data["fma"], data["bp"]
    require(
        "per_track" not in bp and "wider_graph_bridge_witnesses" not in fma,
        "Only aggregate evidence may be published",
    )
    require(
        not fma["reference_labels_or_scores_used"]
        and not bp["key_labels_selected_or_scored"]
        and not fma["dataset_or_pool_changes"]
        and not bp["pool_changes"],
        "metadata-only scope",
    )
    require(
        fma["sources"]["plan"]["sha256"]
        == bp["sources"]["plan"]["sha256"]
        == "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373",
        "cohort identity",
    )
    for role, summary in fma["roles"].items():
        require(
            sum(row["tracks"] for row in summary["bands"].values()) == summary["all"]["tracks"],
            "FMA role duration partition",
        )
        for category in fma["coverage"].values():
            cell = category[role]
            require(
                sum(row["tracks"] for row in cell["bands"].values()) == cell["all"]["tracks"],
                "FMA genre duration partition",
            )
            require(0 <= cell["all"]["components"] <= cell["all"]["tracks"], "FMA component count")
    require(len(fma["all_taxonomy_inventory"]) == 163, "complete FMA taxonomy inventory")
    require(
        fma["roles"]["all_fmak"]["all"]["tracks"] == 5489
        and fma["roles"]["expanded_fit"]["all"]["tracks"] == 4033
        and fma["roles"]["inner_selection"]["all"]["tracks"] == 250,
        "current FMA roles",
    )
    for role, expected in (("fit", 1229), ("inner_selection", 126)):
        summary = bp["roles"][role]
        require(
            summary["tracks"] == summary["workbook_genres_nonempty"] == expected,
            "complete current BP workbook genre coverage",
        )
        require(
            sum(v[role]["tracks"] for v in bp["workbook_genres"].values()) == expected,
            "BP native workbook counts",
        )
        require(sum(summary["preview_bands"].values()) == expected, "BP preview partition")
        require(
            summary["api_json_present"] + summary["api_json_missing"] == expected,
            "BP JSON missingness",
        )
        require(
            summary["actual_preview_seconds_max"] < 121
            and not summary["artist_album_components_verified"],
            "BP preview/group scope",
        )
        for genre in bp["workbook_genres"].values():
            require(
                sum(genre[role]["preview_bands"].values()) == genre[role]["tracks"],
                "BP genre preview partition",
            )
    require(
        bp["store_duration_differs_from_actual_preview_more_than1s"] == 673,
        "store duration must not be substituted for actual previews",
    )
    return data


def load(study: Path = STUDY) -> dict[str, Any]:
    index = json.loads((study / "data/coverage-index.json").read_bytes())
    packed = (study / "data" / index["path"]).read_bytes()
    require(hashlib.sha256(packed).hexdigest() == index["sha256"], "compressed evidence integrity")
    raw = gzip.decompress(packed)
    require(
        hashlib.sha256(raw).hexdigest() == index["decoded_sha256"], "decoded evidence integrity"
    )
    data = verify(json.loads(raw))
    for name, expected in index["aggregate_sha256"].items():
        encoded = json.dumps(data[name], sort_keys=True, separators=(",", ":")).encode()
        require(hashlib.sha256(encoded).hexdigest() == expected, "aggregate source binding")
    return data


def plot(data: dict[str, Any], study: Path = STUDY) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "dj-coverage-v1"}
    )
    fma_names = [
        "Hip-Hop",
        "Rap",
        "Afrobeat",
        "House",
        "House_minus_Chill-out_metadata_slice",
        "Techno",
        "Drum & Bass",
        "Dubstep",
        "named_dance_related_union",
    ]
    bp_names = [
        "Hip-Hop / R&B",
        "Deep House",
        "House",
        "Tech House",
        "Progressive House",
        "Electro House",
        "Techno",
        "Drum & Bass",
        "Dubstep",
    ]
    left = [
        [data["fma"]["coverage"][name]["inner_selection"]["bands"][b]["tracks"] for b in BANDS]
        for name in fma_names
    ]
    right = [
        [
            data["bp"]["workbook_genres"][name]["inner_selection"]["preview_bands"].get(b, 0)
            for b in BANDS
        ]
        for name in bp_names
    ]
    limit = max(v for panel in (left, right) for row in panel for v in row)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), layout="constrained")
    for ax, values, names, title, is_fma in zip(
        axes,
        (left, right),
        (fma_names, bp_names),
        ("FMA selection: published durations", "Beatport selection: actual preview durations"),
        (True, False),
        strict=True,
    ):
        ax.imshow(
            np.asarray(values),
            cmap="Blues",
            vmin=0,
            vmax=limit,
            aspect="auto",
            interpolation="nearest",
        )
        display = [
            n.replace("House_minus_Chill-out_metadata_slice", "House minus Chill-out").replace(
                "named_dance_related_union", "Named dance-tag union"
            )
            for n in names
        ]
        ax.set(
            yticks=range(len(names)),
            yticklabels=display,
            xticks=range(5),
            xticklabels=BANDS,
            title=title,
            xlabel="Duration band (BP 2–<5m cells hug the 120s boundary)",
        )
        for i, name in enumerate(names):
            for j, b in enumerate(BANDS):
                count = values[i][j]
                text = str(count)
                if is_fma:
                    groups = data["fma"]["coverage"][name]["inner_selection"]["bands"][b][
                        "components"
                    ]
                    text += f"/{groups}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if count > limit * 0.55 else "#17334a",
                )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Observed selection coverage: FMA tracks/components; BP tracks only (artist groups unverified)",
        fontsize=11,
    )
    fig.savefig(
        study / "coverage-genre-duration.png",
        dpi=180,
        metadata={"Software": "Drone metadata coverage"},
    )
    fig.savefig(
        study / "coverage-genre-duration.svg",
        metadata={"Date": None, "Creator": "Drone metadata coverage"},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--replay-fma", type=Path)
    parser.add_argument("--replay-bp", type=Path)
    args = parser.parse_args()
    data = load()
    require(
        (args.replay_fma is None) == (args.replay_bp is None), "Both replay aggregates required"
    )
    if args.replay_fma is not None:
        require(
            json.loads(args.replay_fma.read_bytes()) == data["fma"]
            and json.loads(args.replay_bp.read_bytes()) == data["bp"],
            "Metadata replay differs",
        )
    if args.plot:
        plot(data)
    print(
        json.dumps(
            {
                "metadata_only": True,
                "fma_roles": {k: v["all"]["tracks"] for k, v in data["fma"]["roles"].items()},
                "bp_roles": data["bp"]["roles"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
