#!/usr/bin/env bash
set -euo pipefail


PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"


VENV_DIR="$PROJECT_ROOT/venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[ERROR] wrong venv directory: $VENV_DIR" >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"


CONFIG_PATH="$PROJECT_ROOT/config/config.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] config file not found: $CONFIG_PATH" >&2
  exit 1
fi

eval "$(python - <<'PY'
from __future__ import annotations
from pathlib import Path
import yaml

project_root = Path(__file__).resolve().parent
config_path = project_root / 'config' / 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

dataset_name = ((cfg.get('dataset') or {}).get('name')) or 'princeton-nlp/SWE-bench_Lite'
workspace_path = ((cfg.get('workspace') or {}).get('path')) or 'workspace'
preds_dir = (((cfg.get('result') or {}).get('preds') or {}).get('path')) or 'result'
preds_path = (project_root / workspace_path / preds_dir / 'preds.json').resolve()

print(f'DATASET_NAME="{dataset_name}"')
print(f'PREDICTIONS_PATH="{preds_path}"')
PY
)"


if [[ $# -eq 0 ]]; then
  echo "[ERROR] run_id is required" >&2
  echo "Usage: $0 <run_id>" >&2
  exit 1
fi

RUN_ID="$1"

echo "Using dataset: $DATASET_NAME"
echo "Using predictions: $PREDICTIONS_PATH"
echo "Using run_id: $RUN_ID"

# run evaluation
python -m swebench.harness.run_evaluation \
  --dataset_name "$DATASET_NAME" \
  --predictions_path "$PREDICTIONS_PATH" \
  --max_workers 20 \
  --cache_level instance \
  --run_id "$RUN_ID"


