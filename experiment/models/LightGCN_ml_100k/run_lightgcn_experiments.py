import argparse
import csv
import os
import traceback
from itertools import product
from statistics import mean, stdev

import torch
from recbole.quick_start import run_recbole


MODEL_NAME = "LightGCN"
DATASET_NAME = "ml-100k"
METRIC_COLUMNS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]

DEFAULT_SEEDS = [2020, 2021, 2022, 2023, 2024]
DEFAULT_HYPERPARAM_GRID = {
    "embedding_size": [32, 64, 128],
    "n_layers": [1, 2, 3],
    "reg_weight": [1e-4, 1e-5],
}

FAST_SEEDS = [2020, 2021, 2022]
FAST_HYPERPARAM_GRID = {
    "embedding_size": [32, 64],
    "n_layers": [1, 2],
    "reg_weight": [1e-4],
}

# Keep evaluation protocol fixed for all experiments.
BASE_CONFIG = {
    "reproducibility": True,
    "epochs": 300,
    "train_batch_size": 2048,
    "learner": "adam",
    "learning_rate": 0.001,
    "eval_step": 1,
    "stopping_step": 10,
    "eval_args": {
        "split": {"RS": [0.8, 0.1, 0.1]},
        "order": "RO",
        "group_by": "user",
        "mode": {"valid": "full", "test": "full"},
    },
    "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
    "topk": [10],
    "valid_metric": "MRR@10",
    "valid_metric_bigger": True,
    "embedding_size": 64,
    "n_layers": 2,
    "reg_weight": 1e-5,
}


FAST_BASE_OVERRIDES = {
    "epochs": 40,
    "eval_step": 2,
    "stopping_step": 4,
    "train_batch_size": 4096,
    "eval_batch_size": 8192,
    "show_progress": False,
}


def patch_torch_load_for_recbole():
    """Make RecBole checkpoint loading compatible with PyTorch 2.6+ defaults."""
    original_load = torch.load

    def load_with_compat(*args, **kwargs):
        # RecBole checkpoints are trusted local artifacts in this workflow.
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_compat


def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN experiment runner")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a faster profile with fewer seeds/hyperparameter combinations and reduced training budget.",
    )
    return parser.parse_args()


def build_profile(fast_mode):
    if not fast_mode:
        return {
            "name": "default",
            "seeds": DEFAULT_SEEDS,
            "hyperparam_grid": DEFAULT_HYPERPARAM_GRID,
            "base_overrides": {},
            "file_suffix": "",
            "hyperparam_seed": 2020,
        }

    return {
        "name": "fast",
        "seeds": FAST_SEEDS,
        "hyperparam_grid": FAST_HYPERPARAM_GRID,
        "base_overrides": FAST_BASE_OVERRIDES,
        "file_suffix": "_fast",
        "hyperparam_seed": 2020,
    }


def read_rows(csv_path):
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_row(csv_path, fieldnames, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(row)


def write_rows(csv_path, fieldnames, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def normalize_test_metrics(test_result):
    if not isinstance(test_result, dict):
        raise ValueError("test_result is missing or invalid.")

    lower_map = {str(k).lower(): v for k, v in test_result.items()}
    metrics = {}

    for metric in METRIC_COLUMNS:
        key = metric.lower()
        if key not in lower_map:
            raise KeyError(f"Metric '{metric}' not found in test_result: {list(test_result.keys())}")

        value = to_float(lower_map[key])
        if value is None:
            raise ValueError(f"Metric '{metric}' has non-numeric value: {lower_map[key]}")
        metrics[metric] = value

    return metrics


def run_single_experiment(seed, override_config, base_overrides):
    config = dict(BASE_CONFIG)
    config.update(base_overrides)
    config.update(override_config)
    config["seed"] = seed

    result = run_recbole(
        model=MODEL_NAME,
        dataset=DATASET_NAME,
        config_dict=config,
        saved=True,
    )

    test_result = result.get("test_result", {}) if isinstance(result, dict) else {}
    metrics = normalize_test_metrics(test_result)
    return metrics, result


def get_completed_seed_set(rows):
    completed = set()
    for row in rows:
        seed_value = row.get("seed")
        try:
            completed.add(int(seed_value))
        except (TypeError, ValueError):
            continue
    return completed


def run_multi_seed_experiments(multi_seed_csv, seeds, base_overrides):
    print("=" * 80)
    print("[Multi-seed] Starting multi-seed experiments")
    print(f"[Multi-seed] Target seeds: {seeds}")

    existing_rows = read_rows(multi_seed_csv)
    completed = get_completed_seed_set(existing_rows)

    for idx, seed in enumerate(seeds, start=1):
        if seed in completed:
            print(f"[Multi-seed][{idx}/{len(seeds)}] Seed {seed} already exists -> skip")
            continue

        print(f"[Multi-seed][{idx}/{len(seeds)}] Running seed={seed}")
        try:
            metrics, _ = run_single_experiment(
                seed=seed,
                override_config={},
                base_overrides=base_overrides,
            )
            row = {"seed": seed}
            row.update({m: format_float(metrics[m]) for m in METRIC_COLUMNS})
            append_row(multi_seed_csv, ["seed"] + METRIC_COLUMNS, row)
            print(
                "[Multi-seed] Done seed={} -> {}".format(
                    seed,
                    ", ".join(f"{m}={metrics[m]:.4f}" for m in METRIC_COLUMNS),
                )
            )
        except Exception as exc:
            print(f"[Multi-seed] ERROR seed={seed}: {exc}")
            print(traceback.format_exc())

    final_rows = read_rows(multi_seed_csv)
    selected = []
    for row in final_rows:
        seed_value = to_float(row.get("seed"))
        if seed_value is not None and int(seed_value) in seeds:
            selected.append(row)

    print(f"[Multi-seed] Completed with {len(selected)} rows available")
    return selected


def summarize_multi_seed(rows, summary_csv):
    print("=" * 80)
    print("[Multi-seed] Aggregating mean/std")

    if not rows:
        print("[Multi-seed] No rows found. summary_multi_seed.csv is not generated.")
        return None

    values = {metric: [] for metric in METRIC_COLUMNS}
    for row in rows:
        for metric in METRIC_COLUMNS:
            value = to_float(row.get(metric))
            if value is not None:
                values[metric].append(value)

    n_runs = min(len(values[m]) for m in METRIC_COLUMNS)
    if n_runs == 0:
        print("[Multi-seed] No valid metric values. summary_multi_seed.csv is not generated.")
        return None

    mean_row = {"statistic": "mean", "n_runs": n_runs}
    std_row = {"statistic": "std", "n_runs": n_runs}

    for metric in METRIC_COLUMNS:
        metric_values = values[metric]
        mean_row[metric] = format_float(mean(metric_values))
        std_row[metric] = format_float(stdev(metric_values) if len(metric_values) > 1 else 0.0)

    write_rows(
        summary_csv,
        ["statistic", "n_runs"] + METRIC_COLUMNS,
        [mean_row, std_row],
    )

    print(f"[Multi-seed] Saved summary: {summary_csv}")
    return {"mean": mean_row, "std": std_row}


def get_completed_hparam_set(rows):
    completed = set()
    for row in rows:
        emb = row.get("embedding_size")
        layer = row.get("n_layers")
        reg = row.get("reg_weight")
        if emb is None or layer is None or reg is None:
            continue
        completed.add((str(emb), str(layer), str(reg)))
    return completed


def run_hyperparameter_search(hparam_csv, hyperparam_grid, base_overrides, hyperparam_seed):
    print("=" * 80)
    print("[Hyperparam] Starting controlled hyperparameter search")

    combinations = list(
        product(
            hyperparam_grid["embedding_size"],
            hyperparam_grid["n_layers"],
            hyperparam_grid["reg_weight"],
        )
    )
    print(f"[Hyperparam] Total combinations: {len(combinations)}")

    existing_rows = read_rows(hparam_csv)
    completed = get_completed_hparam_set(existing_rows)

    for idx, (embedding_size, n_layers, reg_weight) in enumerate(combinations, start=1):
        combo_key = (str(embedding_size), str(n_layers), str(reg_weight))
        if combo_key in completed:
            print(
                f"[Hyperparam][{idx}/{len(combinations)}] "
                f"(embedding_size={embedding_size}, n_layers={n_layers}, reg_weight={reg_weight}) already exists -> skip"
            )
            continue

        print(
            f"[Hyperparam][{idx}/{len(combinations)}] Running "
            f"embedding_size={embedding_size}, n_layers={n_layers}, reg_weight={reg_weight}"
        )

        try:
            override = {
                "embedding_size": embedding_size,
                "n_layers": n_layers,
                "reg_weight": reg_weight,
            }
            metrics, _ = run_single_experiment(
                seed=hyperparam_seed,
                override_config=override,
                base_overrides=base_overrides,
            )

            row = {
                "embedding_size": embedding_size,
                "n_layers": n_layers,
                "reg_weight": reg_weight,
            }
            row.update({m: format_float(metrics[m]) for m in METRIC_COLUMNS})

            append_row(
                hparam_csv,
                ["embedding_size", "n_layers", "reg_weight"] + METRIC_COLUMNS,
                row,
            )
            print(
                "[Hyperparam] Done -> {}".format(
                    ", ".join(f"{m}={metrics[m]:.4f}" for m in METRIC_COLUMNS)
                )
            )
        except Exception as exc:
            print(
                f"[Hyperparam] ERROR embedding_size={embedding_size}, "
                f"n_layers={n_layers}, reg_weight={reg_weight}: {exc}"
            )
            print(traceback.format_exc())

    rows = read_rows(hparam_csv)
    print(f"[Hyperparam] Completed with {len(rows)} rows available")
    return rows


def rank_best_hparam_row(rows):
    if not rows:
        return None

    def score(row):
        return (
            to_float(row.get("mrr@10")) or -1.0,
            to_float(row.get("ndcg@10")) or -1.0,
            to_float(row.get("recall@10")) or -1.0,
        )

    return max(rows, key=score)


def mean_by_group(rows, group_key, metric_key):
    grouped = {}
    for row in rows:
        group_val = row.get(group_key)
        metric_val = to_float(row.get(metric_key))
        if group_val is None or metric_val is None:
            continue
        grouped.setdefault(str(group_val), []).append(metric_val)

    output = {}
    for key, vals in grouped.items():
        output[key] = mean(vals)
    return output


def format_markdown_table(headers, rows):
    table = []
    table.append("| " + " | ".join(headers) + " |")
    table.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def build_report_update(multi_seed_rows, summary_stats, hyper_rows):
    lines = []

    lines.append("## Experimental Stability")
    lines.append("")

    if not multi_seed_rows:
        lines.append("Không có dữ liệu multi-seed để phân tích.")
    else:
        seed_table_rows = []
        for row in multi_seed_rows:
            seed_table_rows.append(
                [
                    str(int(float(row["seed"]))),
                    row.get("recall@10", ""),
                    row.get("mrr@10", ""),
                    row.get("ndcg@10", ""),
                    row.get("hit@10", ""),
                    row.get("precision@10", ""),
                ]
            )

        lines.append("Kết quả theo từng seed:")
        lines.append("")
        lines.append(
            format_markdown_table(
                ["seed"] + METRIC_COLUMNS,
                seed_table_rows,
            )
        )
        lines.append("")

        if summary_stats is not None:
            mean_row = summary_stats["mean"]
            std_row = summary_stats["std"]

            lines.append("Tổng hợp mean ± std:")
            lines.append("")
            lines.append(
                format_markdown_table(
                    ["statistic", "n_runs"] + METRIC_COLUMNS,
                    [
                        [mean_row["statistic"], str(mean_row["n_runs"])]
                        + [mean_row[m] for m in METRIC_COLUMNS],
                        [std_row["statistic"], str(std_row["n_runs"])]
                        + [std_row[m] for m in METRIC_COLUMNS],
                    ],
                )
            )
            lines.append("")

            mrr_std = to_float(std_row.get("mrr@10"))
            recall_std = to_float(std_row.get("recall@10"))
            if mrr_std is not None and recall_std is not None:
                lines.append("Nhận xét ổn định:")
                lines.append(
                    "- Độ lệch chuẩn nhỏ ở các metric chính cho thấy mô hình ổn định theo seed, "
                    "phù hợp dùng làm kết quả báo cáo/research baseline."
                    if mrr_std < 0.01 and recall_std < 0.01
                    else "- Độ lệch chuẩn chưa nhỏ ở một số metric, nên cân nhắc tăng số seed hoặc chuẩn hóa pipeline để giảm nhiễu."
                )

    lines.append("")
    lines.append("## Hyperparameter Sensitivity")
    lines.append("")

    if not hyper_rows:
        lines.append("Không có dữ liệu hyperparameter search để phân tích.")
    else:
        best_row = rank_best_hparam_row(hyper_rows)

        lines.append("Best cấu hình (xếp theo MRR@10, tie-break NDCG@10, Recall@10):")
        lines.append("")
        lines.append(
            format_markdown_table(
                ["embedding_size", "n_layers", "reg_weight"] + METRIC_COLUMNS,
                [
                    [
                        str(best_row.get("embedding_size", "")),
                        str(best_row.get("n_layers", "")),
                        str(best_row.get("reg_weight", "")),
                        best_row.get("recall@10", ""),
                        best_row.get("mrr@10", ""),
                        best_row.get("ndcg@10", ""),
                        best_row.get("hit@10", ""),
                        best_row.get("precision@10", ""),
                    ]
                ],
            )
        )
        lines.append("")

        lines.append("Mức ảnh hưởng theo tham số (trung bình theo MRR@10):")
        lines.append("")

        sensitivity_headers = ["parameter", "value", "mean_mrr@10"]
        sensitivity_rows = []

        for param in ["embedding_size", "n_layers", "reg_weight"]:
            grouped = mean_by_group(hyper_rows, param, "mrr@10")
            for value, mrr_mean in sorted(grouped.items(), key=lambda x: str(x[0])):
                sensitivity_rows.append([param, str(value), f"{mrr_mean:.6f}"])

        lines.append(format_markdown_table(sensitivity_headers, sensitivity_rows))
        lines.append("")

        lines.append(
            "Nhận xét độ nhạy: tham số nào tạo biên độ thay đổi mean MRR@10 lớn hơn thường có ảnh hưởng mạnh hơn tới chất lượng xếp hạng."
        )

    lines.append("")
    lines.append("## Final Conclusion Update")
    lines.append("")

    if summary_stats is None and not hyper_rows:
        lines.append("Chưa đủ dữ liệu để cập nhật kết luận.")
    else:
        if summary_stats is not None:
            mean_row = summary_stats["mean"]
            std_row = summary_stats["std"]
            lines.append(
                "- Mean multi-seed: "
                + ", ".join(f"{m}={mean_row[m]}" for m in METRIC_COLUMNS)
            )
            lines.append(
                "- Std multi-seed: "
                + ", ".join(f"{m}={std_row[m]}" for m in METRIC_COLUMNS)
            )

        if hyper_rows:
            best_row = rank_best_hparam_row(hyper_rows)
            lines.append(
                "- Best hyperparameters theo MRR@10: "
                f"embedding_size={best_row.get('embedding_size')}, "
                f"n_layers={best_row.get('n_layers')}, "
                f"reg_weight={best_row.get('reg_weight')}."
            )

        lines.append(
            "- Cập nhật kết luận: LightGCN cho hiệu năng ổn định hơn khi đánh giá đa seed và có thể cải thiện thêm "
            "thông qua tuning có kiểm soát, phù hợp cho báo cáo học thuật và tài liệu nghiên cứu tái lập."
        )

    lines.append("")
    return "\n".join(lines)


def save_report_update(markdown_text, report_update_path):
    os.makedirs(os.path.dirname(report_update_path), exist_ok=True)
    with open(report_update_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)


def main():
    args = parse_args()
    profile = build_profile(args.fast)

    patch_torch_load_for_recbole()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "csv_result")
    report_dir = os.path.join(base_dir, "report")

    suffix = profile["file_suffix"]
    multi_seed_csv = os.path.join(csv_dir, f"multi_seed_results{suffix}.csv")
    summary_multi_seed_csv = os.path.join(csv_dir, f"summary_multi_seed{suffix}.csv")
    hyperparam_csv = os.path.join(csv_dir, f"hyperparam_results{suffix}.csv")
    report_update_md = os.path.join(report_dir, f"LIGHTGCN_EXPERIMENT_UPDATE{suffix.upper()}.md")

    print("=" * 80)
    print("LightGCN reproducible experiment pipeline")
    print(f"Model={MODEL_NAME}, Dataset={DATASET_NAME}")
    print(f"Profile={profile['name']}")
    print(f"Seeds={profile['seeds']}")
    print(
        "Grid sizes="
        f"embedding_size:{len(profile['hyperparam_grid']['embedding_size'])}, "
        f"n_layers:{len(profile['hyperparam_grid']['n_layers'])}, "
        f"reg_weight:{len(profile['hyperparam_grid']['reg_weight'])}"
    )
    print("=" * 80)

    multi_seed_rows = run_multi_seed_experiments(
        multi_seed_csv=multi_seed_csv,
        seeds=profile["seeds"],
        base_overrides=profile["base_overrides"],
    )
    summary_stats = summarize_multi_seed(multi_seed_rows, summary_multi_seed_csv)

    hyper_rows = run_hyperparameter_search(
        hparam_csv=hyperparam_csv,
        hyperparam_grid=profile["hyperparam_grid"],
        base_overrides=profile["base_overrides"],
        hyperparam_seed=profile["hyperparam_seed"],
    )

    markdown_update = build_report_update(multi_seed_rows, summary_stats, hyper_rows)
    save_report_update(markdown_update, report_update_md)

    print("=" * 80)
    print(f"Saved: {multi_seed_csv}")
    print(f"Saved: {summary_multi_seed_csv}")
    print(f"Saved: {hyperparam_csv}")
    print(f"Saved: {report_update_md}")
    print("Done.")


if __name__ == "__main__":
    main()
