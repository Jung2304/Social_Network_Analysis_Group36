import csv
import os
import re
from datetime import datetime


PREFERRED_BASE_METRICS = ["recall", "mrr", "ndcg", "hit", "precision"]


def find_latest_log_file(log_dir):
  """Return the newest .log file in log_dir based on modification time."""
  if not os.path.isdir(log_dir):
    return None

  candidates = []
  for filename in os.listdir(log_dir):
    path = os.path.join(log_dir, filename)
    if os.path.isfile(path) and filename.lower().endswith(".log"):
      candidates.append(path)

  if not candidates:
    return None

  return max(candidates, key=os.path.getmtime)


def parse_filename_metadata(log_path):
  """Extract model, dataset, and timestamp from RecBole log filename when possible."""
  filename = os.path.basename(log_path)
  fallback_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  # Expected pattern example:
  # LightGCN-ml-100k-Apr-08-2026_09-58-09-79e0ff.log
  pattern = re.compile(
    r"^(?P<model>[^-]+)-(?P<dataset>.+)-"
    r"(?P<ts>[A-Za-z]{3}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})-"
    r"[^.]+\.log$"
  )
  match = pattern.match(filename)

  if not match:
    return {
      "model": "UnknownModel",
      "dataset": "UnknownDataset",
      "timestamp": fallback_timestamp,
    }

  raw_timestamp = match.group("ts")
  try:
    parsed = datetime.strptime(raw_timestamp, "%b-%d-%Y_%H-%M-%S")
    timestamp = parsed.strftime("%Y-%m-%d %H:%M:%S")
  except ValueError:
    timestamp = fallback_timestamp

  return {
    "model": match.group("model"),
    "dataset": match.group("dataset"),
    "timestamp": timestamp,
  }


def parse_metrics_line(line):
  """Parse all metric@K : value pairs from a line into a dictionary."""
  # Example: recall@10 : 0.1626    mrr@10 : 0.3162
  pairs = re.findall(r"([A-Za-z]+@\d+)\s*:\s*([-+]?\d*\.?\d+)", line)
  metrics = {}
  for key, value in pairs:
    metrics[key.lower()] = float(value)
  return metrics


def extract_history_and_best_epoch(lines):
  """Extract all epoch metrics and best epoch from log lines."""
  epoch_pattern = re.compile(r"epoch\s+(\d+)\s+evaluating", re.IGNORECASE)
  best_epoch_pattern = re.compile(r"best eval result in epoch\s+(\d+)", re.IGNORECASE)

  history = []
  current_epoch = None
  best_epoch = None

  for idx, line in enumerate(lines):
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
      current_epoch = int(epoch_match.group(1))

    best_match = best_epoch_pattern.search(line)
    if best_match:
      best_epoch = int(best_match.group(1))

    if "valid result" in line.lower() and idx + 1 < len(lines):
      metrics = parse_metrics_line(lines[idx + 1])
      if metrics:
        history.append({"epoch": current_epoch, **metrics})

  return history, best_epoch


def choose_metric_columns(sample_metrics):
  """Pick output metric columns dynamically while prioritizing standard ranking metrics."""
  metric_keys = [k for k in sample_metrics.keys() if k != "epoch"]
  selected = []

  for base in PREFERRED_BASE_METRICS:
    candidates = [k for k in metric_keys if k.startswith(base + "@")]
    if candidates:
      # Prefer smaller K when there are multiple variants of the same metric.
      selected.append(min(candidates, key=lambda x: int(x.split("@")[1])))

  for key in sorted(metric_keys):
    if key not in selected:
      selected.append(key)

  return selected


def select_best_row(history, best_epoch):
  """Get the best-epoch row if possible, otherwise fallback to the latest row."""
  if not history:
    return None, None

  if best_epoch is not None:
    for row in history:
      if row.get("epoch") == best_epoch:
        return row, best_epoch

  latest_row = history[-1]
  return latest_row, latest_row.get("epoch")


def get_existing_header(csv_path):
  """Read existing CSV header if file exists and is non-empty."""
  if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
    return None

  with open(csv_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    try:
      return next(reader)
    except StopIteration:
      return None


def append_rows(csv_path, desired_header, rows):
  """Append rows to CSV and create header for new files."""
  if not rows:
    return

  existing_header = get_existing_header(csv_path)
  header = existing_header if existing_header else desired_header

  if existing_header and existing_header != desired_header:
    print(
      f"Warning: header mismatch in '{csv_path}'. "
      "Using existing header to keep append mode safe."
    )

  file_exists = os.path.isfile(csv_path)
  with open(csv_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if not file_exists or os.path.getsize(csv_path) == 0:
      writer.writeheader()

    for row in rows:
      normalized = {col: row.get(col, "") for col in header}
      writer.writerow(normalized)


def read_lines(path):
  with open(path, "r", encoding="utf-8", errors="ignore") as f:
    return f.readlines()


def main():
  base_dir = os.path.dirname(os.path.abspath(__file__))
  log_dir = os.path.join(base_dir, "log", "LightGCN")
  summary_csv_path = os.path.join(base_dir, "summary.csv")
  history_csv_path = os.path.join(base_dir, "history.csv")

  latest_log_path = find_latest_log_file(log_dir)
  if latest_log_path is None:
    print(f"Error: no log file found in '{log_dir}'.")
    return

  lines = read_lines(latest_log_path)
  history, best_epoch = extract_history_and_best_epoch(lines)
  if not history:
    print(f"Error: no valid result blocks found in '{latest_log_path}'.")
    return

  best_row, best_epoch = select_best_row(history, best_epoch)
  if best_row is None:
    print("Error: could not determine best result row.")
    return

  metadata = parse_filename_metadata(latest_log_path)
  metric_columns = choose_metric_columns(best_row)

  summary_header = ["timestamp", "model", "dataset", "best_epoch"] + metric_columns
  history_header = ["timestamp", "epoch"] + metric_columns

  summary_row = {
    "timestamp": metadata["timestamp"],
    "model": metadata["model"],
    "dataset": metadata["dataset"],
    "best_epoch": best_epoch,
  }
  for metric in metric_columns:
    summary_row[metric] = best_row.get(metric, "")

  history_rows = []
  for row in history:
    out = {"timestamp": metadata["timestamp"], "epoch": row.get("epoch")}
    for metric in metric_columns:
      out[metric] = row.get(metric, "")
    history_rows.append(out)

  append_rows(summary_csv_path, summary_header, [summary_row])
  append_rows(history_csv_path, history_header, history_rows)

  print(f"Latest log file: {latest_log_path}")
  print(
    f"Metadata -> model={metadata['model']}, "
    f"dataset={metadata['dataset']}, timestamp={metadata['timestamp']}"
  )
  print(f"Best epoch: {best_epoch}")

  best_parts = []
  for metric in metric_columns:
    value = best_row.get(metric)
    if isinstance(value, float):
      best_parts.append(f"{metric}={value:.4f}")
    else:
      best_parts.append(f"{metric}={value}")
  print("Best result -> " + ", ".join(best_parts))

  print(f"Saved summary row to: {summary_csv_path}")
  print(f"Saved {len(history_rows)} history rows to: {history_csv_path}")


if __name__ == "__main__":
  main()