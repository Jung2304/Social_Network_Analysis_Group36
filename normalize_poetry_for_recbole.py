from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

REQUIRED_COLUMNS = ["user_id", "book_id", "date_added"]
DATE_ADDED_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def load_data(input_path: str, chunksize: int = 500_000) -> pd.DataFrame:
    """Load interactions and keep only required columns."""
    print(f"[load_data] Reading input: {input_path}")
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    chunks: list[pd.DataFrame] = []

    try:
        reader = pd.read_json(
            input_file,
            lines=True,
            chunksize=chunksize,
            dtype={"user_id": "string", "book_id": "string", "date_added": "string"},
        )

        for idx, chunk in enumerate(reader, start=1):
            missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
            if missing:
                raise KeyError(f"Missing required columns: {sorted(missing)}")

            chunks.append(chunk[REQUIRED_COLUMNS].copy())
            print(f"[load_data] Chunk {idx}: +{len(chunk):,} rows")

    except ValueError:
        # Fallback if the file is a regular JSON array instead of line-delimited JSON.
        df = pd.read_json(input_file, lines=False)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")
        chunks = [df[REQUIRED_COLUMNS].copy()]

    if not chunks:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.concat(chunks, ignore_index=True)
    print(f"[load_data] Total interactions loaded: {len(df):,}")
    return df


def filter_year(df: pd.DataFrame, min_year: int = 2007) -> pd.DataFrame:
    """Keep interactions whose date_added year is >= min_year."""
    print(f"[filter_year] Filtering interactions from year >= {min_year}")

    parsed_date = pd.to_datetime(
        df["date_added"],
        format=DATE_ADDED_FORMAT,
        errors="coerce",
        utc=True,
    )
    valid_mask = parsed_date.notna() & (parsed_date.dt.year >= min_year)

    filtered = df.loc[valid_mask, ["user_id", "book_id"]].copy()
    filtered["date_added_dt"] = parsed_date.loc[valid_mask]

    print(
        f"[filter_year] Kept {len(filtered):,}/{len(df):,} interactions "
        f"({len(filtered) / max(len(df), 1):.2%})"
    )
    return filtered


def iterative_k_core(
    df: pd.DataFrame,
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "book_id",
) -> pd.DataFrame:
    """
    Iteratively filter until all users/items have at least k interactions.
    """
    print(f"[iterative_k_core] Starting iterative k-core with k={k}")

    current = df.copy()
    prev_interactions = -1
    iteration = 0

    while True:
        iteration += 1
        before = len(current)

        user_counts = current[user_col].value_counts()
        valid_users = user_counts[user_counts >= k].index
        current = current[current[user_col].isin(valid_users)]

        item_counts = current[item_col].value_counts()
        valid_items = item_counts[item_counts >= k].index
        current = current[current[item_col].isin(valid_items)]

        after = len(current)
        n_users = current[user_col].nunique()
        n_items = current[item_col].nunique()

        print(
            f"[iterative_k_core] Iter {iteration}: "
            f"Users={n_users:,}, Items={n_items:,}, Interactions={after:,} "
            f"(removed {before - after:,})"
        )

        if after == 0:
            print("[iterative_k_core] Dataset became empty during filtering.")
            break

        if after == prev_interactions:
            print(f"[iterative_k_core] Converged after {iteration} iterations.")
            break

        prev_interactions = after

    if len(current) > 0:
        min_user_degree = int(current[user_col].value_counts().min())
        min_item_degree = int(current[item_col].value_counts().min())
        print(
            f"[iterative_k_core] Final min degrees -> "
            f"user: {min_user_degree}, item: {min_item_degree}"
        )

    return current.reset_index(drop=True)


def save_to_inter(df: pd.DataFrame, output_path: str) -> None:
    """Save dataframe to RecBole .inter format (TSV with type annotations)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = df[["user_id", "book_id", "timestamp"]].rename(columns={"book_id": "item_id"})
    out_df["user_id"] = out_df["user_id"].astype("string")
    out_df["item_id"] = out_df["item_id"].astype("string")
    out_df["timestamp"] = out_df["timestamp"].astype("float64")

    out_df.to_csv(
        out_path,
        sep="\t",
        index=False,
        header=["user_id:token", "item_id:token", "timestamp:float"],
    )

    print(f"[save_to_inter] Saved file: {out_path}")


def print_final_stats(df: pd.DataFrame) -> None:
    users = df["user_id"].nunique()
    items = df["book_id"].nunique()
    interactions = len(df)
    density = interactions / (users * items) if users > 0 and items > 0 else 0.0

    print("[final_stats]")
    print(f"  Users: {users:,}")
    print(f"  Items: {items:,}")
    print(f"  Interactions: {interactions:,}")
    print(f"  Density: {density:.8f}")

def save_config_yaml(dataset_name: str, output_dir: str):
    """Tạo file cấu hình .yaml cơ bản để RecBole nhận diện dữ liệu."""
    config = {
        # Config về dữ liệu (Đây là phần Quân cần cam kết với bạn cùng nhóm)
        'data_path': 'dataset/',
        'dataset': dataset_name,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp']
        },
        
        # Config về cách chia dữ liệu (Dựa trên logic timestamp Quân đã làm)
        'eval_args': {
            'split': {'RS': [8, 1, 1]}, # Chia 8:1:1
            'group_by': 'user',
            'order': 'TO'              # TO = Temporal Order (Sắp xếp theo thời gian)
        },
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'topk': [10]
    }

    yaml_path = Path(output_dir) / f"{dataset_name}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"[save_config_yaml] Created config file: {yaml_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Goodreads Poetry interactions into RecBole .inter format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="goodreads_interactions_poetry.json",
        help="Path to raw Goodreads interactions file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/poetry/poetry.inter",
        help="Output path for RecBole .inter file",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2007,
        help="Keep interactions with date_added year >= min-year",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K value for iterative user-item k-core filtering",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Chunk size when reading line-delimited JSON",
    )

    args = parser.parse_args()

    df = load_data(args.input, chunksize=args.chunksize)
    df = filter_year(df, min_year=args.min_year)
    df = iterative_k_core(df, k=args.k)

    if df.empty:
        raise ValueError("No data left after filtering. Try reducing constraints.")

    # Convert timezone-aware datetime to Unix seconds robustly across us/ns backends.
    df["timestamp"] = (
        df["date_added_dt"]
        .dt.tz_localize(None)
        .astype("datetime64[s]")
        .astype("int64")
        .astype("float64")
    )
    df = df.sort_values("timestamp", ascending=True, kind="mergesort").reset_index(drop=True)

    print_final_stats(df)
    save_to_inter(df, args.output)

    # Tự động tạo file config đi kèm
    dataset_name = Path(args.output).parent.name # Lấy tên 'poetry' từ path
    save_config_yaml(dataset_name, Path(args.output).parent)
    
if __name__ == "__main__":
    main()
