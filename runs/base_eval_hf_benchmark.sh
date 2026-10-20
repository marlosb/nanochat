#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=python
EVAL=core
MAX_PER_TASK=-1
DEVICE_TYPE=
STOP_ON_ERROR=0
DRY_RUN=0
OVERALL_RC=0
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
Usage: bash runs/base_eval_hf_benchmark.sh [options]

Runs Hugging Face models sequentially, using CORE and all examples by default.

Options:
  --python <exe>              Python executable. Default: python
  --eval <modes>              Evaluations: core,bpb,sample. Default: core
  --max-per-task <n>          Max examples per CORE task. Default: -1 (all)
  --device-type <type>        cuda|cpu|mps (default: autodetect)
  --nanochat-base-dir <path>  Sets NANOCHAT_BASE_DIR for this run
  --models "<m1 m2 ...>"      Space- or comma-separated HF model list
  --stop-on-error            Stop on the first failing model
  --dry-run                  Print commands without executing
  -h, --help                 Show this help

ArthurFriedenreich and CharlesMiller execute custom code from their HF repositories.
EOF
}

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --python|--eval|--max-per-task|--device-type|--nanochat-base-dir|--models)
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
for model in "${MODELS[@]}"; do
    echo "[$(date '+%F %T')] model running - $model"
    cmd=("$PYTHON" -m scripts.base_eval --eval "$EVAL" --hf-path "$model"
         --max-per-task "$MAX_PER_TASK" --append-report)
    case "$model" in
        marlosb/ArthurFriedenreich|marlosb/CharlesMiller)
            cmd+=(--trust-remote-code)
            ;;
    esac
    if [[ -n "$DEVICE_TYPE" ]]; then
        cmd+=(--device-type "$DEVICE_TYPE")
    fi

    rc=0
    if ((DRY_RUN)); then
        printf '[dry-run]'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}" || rc=$?
    fi

    if ((rc != 0)); then
        echo "[$(date '+%F %T')] model failed - $model (exit $rc)" >&2
        if ((OVERALL_RC == 0)); then
            OVERALL_RC=$rc
        fi
        if ((STOP_ON_ERROR)); then
            exit "$rc"
        fi
    else
        echo "[$(date '+%F %T')] model done - $model"
    fi
done
exit "$OVERALL_RC"
