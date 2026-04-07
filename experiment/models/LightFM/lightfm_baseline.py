from __future__ import annotations

import argparse
import csv
import math
from itertools import product
from pathlib import Path

import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k, recall_at_k
from scipy.sparse import coo_matrix

CSV_COLUMNS = [
    "loss",
    "precision@10",
    "recall@10",
    "mrr@10",
    "ndcg@10",
    "hit@10",
    "epochs",
    "components",
    "learning_rate",
    "k",
    "random_state",
]


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def load_data(min_rating: float = 4.0) -> tuple[coo_matrix, coo_matrix]:
    data = fetch_movielens(min_rating=min_rating)
    train_interactions = data["train"].tocoo().astype(np.float32)
    test_interactions = data["test"].tocoo().astype(np.float32)

    print("Train shape:", train_interactions.shape, flush=True)
    print("Test shape :", test_interactions.shape, flush=True)
    print("Train nnz  :", train_interactions.nnz, flush=True)
    print("Test nnz   :", test_interactions.nnz, flush=True)

    return train_interactions, test_interactions


def train_model(
    train_interactions: coo_matrix,
    no_components: int,
    learning_rate: float,
    epochs: int,
    num_threads: int,
    random_state: int,
) -> LightFM:
    model = LightFM(
        no_components=no_components,
        learning_rate=learning_rate,
        loss="logistic",
        random_state=random_state,
    )
    model.fit(train_interactions, epochs=epochs, num_threads=num_threads, verbose=False)
    return model


def evaluate_model(
    model: LightFM,
    train_interactions: coo_matrix,
    test_interactions: coo_matrix,
    k: int,
    num_threads: int,
) -> dict[str, float]:
    precision_k = precision_at_k(
        model,
        test_interactions,
        train_interactions=train_interactions,
        k=k,
        num_threads=num_threads,
    ).mean()
    recall_k = recall_at_k(
        model,
        test_interactions,
        train_interactions=train_interactions,
        k=k,
        num_threads=num_threads,
    ).mean()

    train_csr = train_interactions.tocsr()
    test_csr = test_interactions.tocsr()
    num_users, num_items = train_csr.shape
    item_ids = np.arange(num_items)
    top_k = min(k, num_items)

    rr_scores: list[float] = []
    ndcg_scores: list[float] = []
    hit_scores: list[float] = []

    for user_id in range(num_users):
        test_items = test_csr.indices[test_csr.indptr[user_id] : test_csr.indptr[user_id + 1]]
        if test_items.size == 0:
            continue

        scores = model.predict(user_id, item_ids, num_threads=num_threads)
        seen_items = train_csr.indices[train_csr.indptr[user_id] : train_csr.indptr[user_id + 1]]
        scores[seen_items] = -np.inf

        candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_items = candidate_idx[np.argsort(-scores[candidate_idx])]
        rel = np.isin(top_items, test_items, assume_unique=False)

        hit_scores.append(float(np.any(rel)))
        if np.any(rel):
            hit_positions = np.flatnonzero(rel) + 1
            rr_scores.append(1.0 / int(hit_positions[0]))
            dcg = float(np.sum([1.0 / math.log2(rank + 1) for rank in hit_positions]))
        else:
            rr_scores.append(0.0)
            dcg = 0.0

        ideal_hits = min(test_items.size, top_k)
        idcg = float(np.sum([1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)]))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "precision@10": float(precision_k),
        "recall@10": float(recall_k),
        "mrr@10": float(np.mean(rr_scores)) if rr_scores else 0.0,
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "hit@10": float(np.mean(hit_scores)) if hit_scores else 0.0,
    }


def append_result(csv_path: Path, row: dict[str, float | int | str]) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_one_config(
    train_interactions: coo_matrix,
    test_interactions: coo_matrix,
    epochs: int,
    components: int,
    learning_rate: float,
    k: int,
    num_threads: int,
    random_state: int,
) -> dict[str, float | int | str]:
    model = train_model(
        train_interactions=train_interactions,
        no_components=components,
        learning_rate=learning_rate,
        epochs=epochs,
        num_threads=num_threads,
        random_state=random_state,
    )
    metrics = evaluate_model(
        model=model,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        k=k,
        num_threads=num_threads,
    )
    return {
        "loss": "logistic",
        "precision@10": metrics["precision@10"],
        "recall@10": metrics["recall@10"],
        "mrr@10": metrics["mrr@10"],
        "ndcg@10": metrics["ndcg@10"],
        "hit@10": metrics["hit@10"],
        "epochs": epochs,
        "components": components,
        "learning_rate": learning_rate,
        "k": k,
        "random_state": random_state,
    }


def run_single(args: argparse.Namespace) -> None:
    csv_path = Path(args.results_csv).resolve()
    train_interactions, test_interactions = load_data()
    row = run_one_config(
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        epochs=args.epochs,
        components=args.components,
        learning_rate=args.learning_rate,
        k=args.k,
        num_threads=args.num_threads,
        random_state=args.random_state,
    )
    append_result(csv_path, row)
    print("SINGLE RUN END", flush=True)
    print(
        " | ".join(
            [
                f"precision@10={row['precision@10']:.4f}",
                f"recall@10={row['recall@10']:.4f}",
                f"mrr@10={row['mrr@10']:.4f}",
                f"ndcg@10={row['ndcg@10']:.4f}",
                f"hit@10={row['hit@10']:.4f}",
            ]
        ),
        flush=True,
    )


def run_grid_search(args: argparse.Namespace) -> None:
    csv_path = Path(args.results_csv).resolve()
    epochs_grid = _parse_int_list(args.grid_epochs)
    components_grid = _parse_int_list(args.grid_components)
    lr_grid = _parse_float_list(args.grid_learning_rates)
    candidates = list(product(epochs_grid, components_grid, lr_grid))

    if args.max_grid_runs > 0:
        candidates = candidates[: args.max_grid_runs]

    print(f"GRID SEARCH CANDIDATES: {len(candidates)}", flush=True)
    print(f"RESULT CSV PATH: {csv_path}", flush=True)

    train_interactions, test_interactions = load_data()
    metric_key = args.grid_selection_metric
    best_row: dict[str, float | int | str] | None = None

    for idx, (epochs, components, learning_rate) in enumerate(candidates, start=1):
        print(
            f"GRID RUN {idx}/{len(candidates)} | "
            f"epochs={epochs}, components={components}, lr={learning_rate}",
            flush=True,
        )
        row = run_one_config(
            train_interactions=train_interactions,
            test_interactions=test_interactions,
            epochs=epochs,
            components=components,
            learning_rate=learning_rate,
            k=args.k,
            num_threads=args.num_threads,
            random_state=args.random_state,
        )
        append_result(csv_path, row)

        if best_row is None or float(row[metric_key]) > float(best_row[metric_key]):
            best_row = row

    if best_row is None:
        print("GRID SEARCH END | no runs", flush=True)
        return

    print("GRID SEARCH BEST RESULT", flush=True)
    print(
        " | ".join(
            [
                f"precision@10={best_row['precision@10']:.4f}",
                f"recall@10={best_row['recall@10']:.4f}",
                f"mrr@10={best_row['mrr@10']:.4f}",
                f"ndcg@10={best_row['ndcg@10']:.4f}",
                f"hit@10={best_row['hit@10']:.4f}",
                f"epochs={best_row['epochs']}",
                f"components={best_row['components']}",
                f"learning_rate={best_row['learning_rate']}",
            ]
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact LightFM logistic experiments")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--components", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-csv", default="lightfm_experiment_results.csv")

    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--grid-epochs", default="10,20")
    parser.add_argument("--grid-components", default="20,30")
    parser.add_argument("--grid-learning-rates", default="0.03,0.05")
    parser.add_argument("--max-grid-runs", type=int, default=8)
    parser.add_argument(
        "--grid-selection-metric",
        choices=["precision@10", "recall@10", "mrr@10", "ndcg@10", "hit@10"],
        default="recall@10",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.grid_search:
        run_grid_search(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
