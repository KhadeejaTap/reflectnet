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


def filter_matte_nonmetal(df):
    """
    Filter to instances that are matte + non-metal, based on tags.json fields
    flattened into the metadata parquet. Assumes columns 'glossiness' and
    'metallic_hint' exist per-row (constant within an instance_id).
    """
    mask = (df["glossiness"] == "matte") & (df["metallic_hint"] == "non-metal")
    kept = df[mask]
    dropped = df[~mask]
    return kept, dropped


def print_filter_summary(df, kept, dropped):
    total_instances = df["instance_id"].nunique()
    kept_instances = kept["instance_id"].nunique()
    dropped_instances = dropped["instance_id"].nunique()

    total_models = df["model_name"].nunique()
    kept_models = kept["model_name"].nunique()
    dropped_models = set(df["model_name"].unique()) - set(kept["model_name"].unique())

    print("\n--- Matte / Non-metal filter ---")
    print(f"Instances: {kept_instances} kept / {dropped_instances} dropped / {total_instances} total")
    print(f"Models with >=1 surviving instance: {kept_models} / {total_models}")
    if dropped_models:
        print(f"Models with ZERO surviving instances: {sorted(dropped_models)}")

    print("\nSurviving instances per model:")
    print(kept.groupby("model_name")["instance_id"].nunique().sort_values(ascending=False))


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

    kept, dropped = filter_matte_nonmetal(df)
    print_filter_summary(df, kept, dropped)
