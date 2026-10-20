#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=python
EVAL=core
MAX_PER_TASK=-1
DEVICE_TYPE=cuda
OUTPUT_DIR=
STOP_ON_ERROR=0
DRY_RUN=0
OVERALL_RC=0
STOP_LAUNCHING=0
MODELS=(
    openai-community/gpt2
    Polygl0t/Tucano2-0.6B-Base
    Polygl0t/Tucano2-qwen-1.5B-Base
    TucanoBR/Tucano-630m
    TucanoBR/Tucano-1b1
    marlosb/ArthurFriedenreich
    marlosb/CharlesMiller
)

usage() {
    cat <<'EOF'
Usage: bash runs/base_eval_hf_benchmark_parallel.sh [options]

Runs up to two Hugging Face models concurrently on the same GPU.
Uses CORE and all examples by default. Requires Bash 4 or newer.

Options:
  --python <exe>              Python executable. Default: python
  --eval <modes>              Evaluations: core,bpb,sample. Default: core
  --max-per-task <n>          Max examples per CORE task. Default: -1 (all)
  --device-type <type>        cuda|cpu|mps. Default: cuda
  --nanochat-base-dir <path>  Shared evaluation data directory
  --output-dir <path>         Output root. Default: <base-dir>/parallel_eval
  --models "<m1 m2 ...>"      Space- or comma-separated HF model list
  --stop-on-error            Stop scheduling after a failure; finish active models
  --dry-run                  Print commands without executing or creating outputs
  -h, --help                 Show this help

Each model gets its own numbered output directory, CSV, report, and evaluation.log.
HF weights and evaluation data remain shared. Existing model outputs are overwritten.
ArthurFriedenreich and CharlesMiller execute custom code from their HF repositories.
EOF
}

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --python|--eval|--max-per-task|--device-type|--nanochat-base-dir|--output-dir|--models)
            if (($# < 2)) || [[ -z "$2" ]]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            case "$1" in
                --python) PYTHON="$2" ;;
                --eval) EVAL="$2" ;;
                --max-per-task) MAX_PER_TASK="$2" ;;
                --device-type) DEVICE_TYPE="$2" ;;
                --nanochat-base-dir) export NANOCHAT_BASE_DIR="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --models)
                    model_list="${2//,/ }"
                    read -r -a MODELS <<< "$model_list"
                    if ((${#MODELS[@]} == 0)); then
                        echo "No models specified" >&2
                        exit 1
                    fi
                    ;;
            esac
            shift 2
            ;;
        --stop-on-error)
            STOP_ON_ERROR=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

cd -- "$REPO_ROOT"
OUTPUT_DIR="${OUTPUT_DIR:-${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}/parallel_eval}"
declare -A ACTIVE=()
NEXT=0

cleanup() {
    local pid
    for pid in "${!ACTIVE[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${!ACTIVE[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while ((NEXT < ${#MODELS[@]} || ${#ACTIVE[@]} > 0)); do
    # Reap every completed child before filling either slot.
    for pid in "${!ACTIVE[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rc=0
            wait "$pid" || rc=$?
            model="${ACTIVE[$pid]}"
            unset 'ACTIVE[$pid]'
            if ((rc != 0)); then
                echo "[$(date '+%F %T')] model failed - $model (exit $rc)" >&2
                if ((OVERALL_RC == 0)); then
                    OVERALL_RC=$rc
                fi
                if ((STOP_ON_ERROR)); then
                    STOP_LAUNCHING=1
                fi
            else
                echo "[$(date '+%F %T')] model done - $model"
            fi
        fi
    done

    if ((STOP_LAUNCHING)); then
        NEXT=${#MODELS[@]}
    fi
    while ((NEXT < ${#MODELS[@]} && ${#ACTIVE[@]} < 2)); do
        model="${MODELS[$NEXT]}"
        slug="${model//\//-}"
        model_output="$OUTPUT_DIR/$(printf '%02d' "$((NEXT + 1))")-$slug"
        cmd=("$PYTHON" -u -m scripts.base_eval --eval "$EVAL" --hf-path "$model"
             --max-per-task "$MAX_PER_TASK" --device-type "$DEVICE_TYPE"
             --output-dir "$model_output")
        case "$model" in
            marlosb/ArthurFriedenreich|marlosb/CharlesMiller)
                cmd+=(--trust-remote-code)
                ;;
        esac
        if ((DRY_RUN)); then
            printf '[dry-run]'
            printf ' %q' "${cmd[@]}"
            printf '\n'
        else
            mkdir -p -- "$model_output"
            echo "[$(date '+%F %T')] model running - $model | log: $model_output/evaluation.log"
            "${cmd[@]}" > "$model_output/evaluation.log" 2>&1 &
            ACTIVE[$!]="$model"
        fi
        NEXT=$((NEXT + 1))
    done
    if ((${#ACTIVE[@]} > 0)); then
        sleep 0.2
    fi
done
exit "$OVERALL_RC"
