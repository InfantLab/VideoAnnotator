"""Do the tools see what the parents saw? The BabyJokes tutorial's analysis.

Joins VideoAnnotator's results table to the parents' ratings in the BabyJokes
dataset (Zenodo 10.5281/zenodo.23913806, CC BY 4.0), then asks:

1. When the parent said the baby laughed, did the baby's face smile?
   (a 2 x 2 table and Cohen's kappa, for two measures of a smile)
2. The study's own question: does laughing fall off from the first
   repetition of a game to the third, and does that depend on the game
   or the baby's age? (in the parents' answers, and in the tools')

Usage:
    python analysis.py RESULTS.csv babyjokes-public-ratings.csv \\
        babyjokes-public-clips.csv [--out by_clip.csv]

RESULTS.csv is the results table exported from the BabyJokes project
(Export > Results table, CSV). Needs pandas only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# A smile here is the baby's lips pulled up at the corners: OpenFace 3's
# AU12 (lip corner puller), present when the model gives it more than an
# even chance. DeepFace's "most likely expression is happy" is the second
# measure. A clip counts as a smile if at least this share of the baby's
# sampled faces smiled. Change it and see how the agreement moves.
SMILE_SHARE = 0.25
AU12 = "AU12_Lip_Corner_Puller"

# Person tracking follows each person through a clip but doesn't say who is
# who. In these games the baby is the smaller person on screen. Tracks seen
# in under this share of the clip's frames with people are passers-by or
# brief mistakes, so they can't be the baby.
MIN_TRACK_SHARE = 0.2


def video_name(file: pd.Series) -> pd.Series:
    """`public/X.mp4` in the dataset's CSVs is video `X` in the table."""
    return file.str.removeprefix("public/").str.removesuffix(".mp4")


def load(
    results: Path, ratings: Path, clips: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(results, low_memory=False)
    rated = pd.read_csv(ratings).merge(
        pd.read_csv(clips)[["file", "age_days"]], on="file"
    )
    rated["video"] = video_name(rated["file"])
    return table, rated


def baby_tracks(table: pd.DataFrame) -> pd.DataFrame:
    """Each clip's baby: the smaller person, of two or more seen often enough.

    With only one person found it is usually the parent: a baby lying down,
    held, or half hidden in a seat often isn't detected as a person at all.
    Those clips get no baby rather than the parent's face."""
    people = table[table["measure"] == "person"].assign(
        area=lambda d: d["width"] * d["height"]
    )
    tracks = people.groupby(["video", "track"]).agg(
        area=("area", "median"), frames=("frame", "nunique")
    )
    tracks["share"] = tracks["frames"] / tracks.groupby("video")["frames"].transform(
        "max"
    )
    seen = tracks[tracks["share"] >= MIN_TRACK_SHARE].reset_index()
    seen = seen[seen.groupby("video")["track"].transform("size") >= 2]
    return seen.sort_values("area").groupby("video").head(1)[["video", "track"]]


def smiles(table: pd.DataFrame) -> pd.DataFrame:
    """Per clip, the share of the baby's faces that smiled, by each measure."""
    baby = table.merge(baby_tracks(table), on=["video", "track"])
    au12 = baby[(baby["measure"] == "action_unit_presence") & (baby["label"] == AU12)]
    deepface = baby[
        (baby["pipeline"] == "face_analysis") & (baby["measure"] == "emotion")
    ]
    return pd.DataFrame(
        {
            "au12_share": au12.groupby("video")["value"].mean(),
            "happy_share": deepface.assign(happy=deepface["label"] == "happy")
            .groupby("video")["happy"]
            .mean(),
        }
    ).reset_index()


def last_take(rated: pd.DataFrame) -> pd.DataFrame:
    """One clip per repetition: the parent rated each repetition once, so
    every take of it carries the same answer and would count it twice."""
    keys = ["child_id", "session", "joke", "repetition"]
    return rated.sort_values("take").groupby(keys).tail(1)


def kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two yes/no ratings of the same clips."""
    observed = (a == b).mean()
    expected = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def agreement(by_clip: pd.DataFrame, measure: str, name: str) -> None:
    rated = by_clip.dropna(subset=[measure])
    if rated.empty:
        print(f"\n{name}: not in these results")
        return
    tool = rated[measure] >= SMILE_SHARE
    parent = rated["parent_laughed"] == "Yes"
    print(f"\n{name}: smiled in at least {SMILE_SHARE:.0%} of the baby's faces")
    print(
        pd.crosstab(
            parent.map({True: "laughed", False: "didn't"}).rename("parent said"),
            tool.map({True: "smiled", False: "didn't"}).rename("tool saw"),
        )
    )
    print(
        f"agree on {(tool == parent).mean():.0%} of {len(rated)} repetitions; "
        f"kappa {kappa(tool, parent):.2f}"
    )


def by_repetition(by_clip: pd.DataFrame) -> None:
    laughed = by_clip.assign(
        parent=by_clip["parent_laughed"] == "Yes",
        smiled=by_clip["au12_share"] >= SMILE_SHARE,
        age_months=pd.cut(
            by_clip["age_days"] / 30.44,
            [0, 6, 12, 18, 40],
            right=False,
            labels=["under 6", "6-11", "12-17", "18 and over"],
        ),
    )
    print("\nShare of repetitions the parent said the baby laughed at:")
    print(
        laughed.pivot_table(
            index="joke", columns="repetition", values="parent", aggfunc="mean"
        ).round(2)
    )
    print("\n... and the tool saw a smile (AU12):")
    print(
        laughed.pivot_table(
            index="joke", columns="repetition", values="smiled", aggfunc="mean"
        ).round(2)
    )
    print("\nBy age (months), parent and tool:")
    print(
        laughed.groupby("age_months", observed=True)[["parent", "smiled"]]
        .agg(["mean", "size"])
        .round(2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="The exported results table (CSV)")
    parser.add_argument("ratings", type=Path, help="babyjokes-public-ratings.csv")
    parser.add_argument("clips", type=Path, help="babyjokes-public-clips.csv")
    parser.add_argument(
        "--out", type=Path, help="Also save one row per repetition here"
    )
    args = parser.parse_args()

    table, rated = load(args.results, args.ratings, args.clips)
    by_clip = last_take(rated).merge(smiles(table), on="video", how="left")
    without = by_clip["au12_share"].isna() & by_clip["happy_share"].isna()
    print(
        f"{len(by_clip)} repetitions rated by a parent; "
        f"{(~without).sum()} with the baby's face in the results"
    )

    agreement(by_clip, "au12_share", "OpenFace 3, AU12")
    agreement(by_clip, "happy_share", "DeepFace, happy")
    by_repetition(by_clip[~without])

    if args.out:
        by_clip.to_csv(args.out, index=False)
        print(f"\nOne row per repetition: {args.out}")


if __name__ == "__main__":
    main()
