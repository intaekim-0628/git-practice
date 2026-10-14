#!/usr/bin/env bash
# Native MXFP4 + BnB-trained attention LoRA, matching the user's existing workflow.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-/home/sherpa/gpt-oss-120b}"
LORA_PATH="${LORA_PATH:-$PACKAGE_DIR/outputs/sql-tuner/best_adapter}"
SERVED_MODEL="${SERVED_MODEL:-gpt-oss-120b-sql}"
BASE_SERVED_MODEL="${BASE_SERVED_MODEL:-gpt-oss-120b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8086}"
PROFILE="${PROFILE:-stable}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
ENABLE_TOOLS="${ENABLE_TOOLS:-0}"
[[ "$SERVED_MODEL" != "$BASE_SERVED_MODEL" ]] || { echo 'Base and adapter API names must differ' >&2; exit 2; }
[[ "$ENABLE_TOOLS" == 0 || "$ENABLE_TOOLS" == 1 ]] || { echo 'ENABLE_TOOLS must be 0 or 1' >&2; exit 2; }
"$PYTHON_BIN" "$SCRIPT_DIR/verify_adapter_compat.py" --adapter "$LORA_PATH" --native-base "$BASE_MODEL"
RANK="$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["r"])' "$LORA_PATH/adapter_config.json")"
MAX_LORA_RANK="${MAX_LORA_RANK:-$RANK}"
[[ "$MAX_LORA_RANK" =~ ^[0-9]+$ && "$MAX_LORA_RANK" -ge "$RANK" ]] || { echo 'MAX_LORA_RANK must cover adapter rank' >&2; exit 2; }
case "$PROFILE" in
  stable) DEFAULT_LEN=8192; DEFAULT_SEQS=2; DEFAULT_BATCH=8192; DEFAULT_MEMORY=0.90 ;;
  throughput) DEFAULT_LEN=16384; DEFAULT_SEQS=4; DEFAULT_BATCH=8192; DEFAULT_MEMORY=0.92 ;;
  long-context) DEFAULT_LEN=65536; DEFAULT_SEQS=4; DEFAULT_BATCH=8192; DEFAULT_MEMORY=0.90 ;;
  *) echo 'PROFILE must be stable, throughput, or long-context' >&2; exit 2 ;;
esac
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_LEN}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$DEFAULT_SEQS}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$DEFAULT_BATCH}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$DEFAULT_MEMORY}"
args=(serve "$BASE_MODEL" --host "$HOST" --port "$PORT"
  --served-model-name "$BASE_SERVED_MODEL" --tensor-parallel-size 1 --dtype auto
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --enable-chunked-prefill --no-enable-prefix-caching --enable-lora --max-loras 1
  --max-lora-rank "$MAX_LORA_RANK" --lora-dtype bfloat16 --lora-modules "$SERVED_MODEL=$LORA_PATH")
if [[ "$ENABLE_TOOLS" == 1 ]]; then args+=(--enable-auto-tool-choice --tool-call-parser openai); fi
if [[ "$DRY_RUN" == 1 ]]; then
  # No secret values are included in this printed command.
  printf 'vllm'; printf ' %q' "${args[@]}"; printf '\n'; exit 0
fi
command -v vllm >/dev/null || { echo 'Activate the existing working vLLM environment first' >&2; exit 2; }
# Confirm all flags against the installed release, without silently dropping settings.
if ! help_text="$(vllm serve --help=all 2>/dev/null)"; then
  help_text="$(vllm serve --help)"
fi
for arg in "${args[@]}"; do
  if [[ "$arg" == --* ]] && ! "$PYTHON_BIN" -c 'import sys;sys.exit(0 if sys.argv[1] in sys.stdin.read() else 1)' "$arg" <<< "$help_text"; then
    echo "Installed vLLM does not advertise $arg; check this release's CLI before starting" >&2; exit 2
  fi
done
# vLLM reads VLLM_API_KEY from its environment, so the key is not echoed in argv.
exec vllm "${args[@]}"
