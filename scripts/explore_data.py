import pandas as pd
from pathlib import Path

METADATA_DIR = Path(__file__).resolve().parent.parent / "data" / "metadata" / "parts" / "train"
PATTERN = "Indoor_1_beds_*_A.parquet"


def load_metadata():
    files = sorted(METADATA_DIR.glob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {METADATA_DIR} matching {PATTERN}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def print_summary(df):
    print(f"Total frames: {len(df)}")
    print(f"Releases: {sorted(df['release_name'].unique())}")
    print(f"Unique models: {df['model_name'].nunique()}")
    print(f"Unique instances: {df['instance_id'].nunique()}")
    print()
    summary = df.groupby("release_name").agg(
        frames=("sample_id", "count"),
        models=("model_name", "nunique"),
        instances=("instance_id", "nunique"),
    )
    print(summary)


def check_camera_paths(df):
    # Compare transform_matrix per frame_id across different instance_ids.
    # If cameras are shared, the same frame_id should have identical transform_matrix
    # regardless of instance_id.
    sample = df[df["frame_id"] == 0]
    unique_transforms = sample["transform_matrix"].apply(lambda m: tuple(map(tuple, m))).nunique()
    print(f"\nAt frame_id=0: {len(sample)} rows, {unique_transforms} unique transform_matrix values")
    if unique_transforms == 1:
        print("-> All instances share the same camera trajectory (identical transform per frame_id).")
    else:
        print("-> Camera transforms differ across instances (not shared).")


if __name__ == "__main__":
    df = load_metadata()
    print_summary(df)
    check_camera_paths(df)
