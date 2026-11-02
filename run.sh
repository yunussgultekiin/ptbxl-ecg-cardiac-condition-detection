#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ARCHITECTURES=(cnn1d fcn resnet1d xresnet1d se_resnet1d inception1d tcn cnn_bilstm transformer1d s4d)
AUGMENTATIONS=(crop mixup cutmix rotation time_mask)
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

usage() {
    echo "usage: ./run.sh prepare | all | <architecture> | augmentation <architecture> [augmentation]"
    echo "architectures: ${ARCHITECTURES[*]}"
    echo "augmentations: ${AUGMENTATIONS[*]}"
    exit 1
}

require_cuda() {
    if ! command -v nvidia-smi > /dev/null || ! nvidia-smi > /dev/null 2>&1; then
        echo "nvidia-smi failed: no CUDA GPU available, training stopped" >&2
        exit 1
    fi
    if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "PyTorch cannot access CUDA, training stopped" >&2
        exit 1
    fi
    echo "CUDA check passed"
}

require_normalization() {
    if ! python -c "import sys; from src.config import ProjectPaths; from src.data import NormalizationStore; sys.exit(0 if NormalizationStore(ProjectPaths()).is_current() else 1)"; then
        echo "normalization statistics are missing or outdated, run ./run.sh prepare, training stopped" >&2
        exit 1
    fi
    echo "normalization check passed"
}

prepare() {
    python -m src.data
}

train() {
    echo "=== training $1 ${2:-} ==="
    python -m src.training --architecture "$1" ${2:+--augmentation "$2"}
}

report() {
    local total=$1 label=$2
    shift 2
    echo "=== finished: $(( total - $# ))/$total $label succeeded ==="
    if (( $# > 0 )); then
        echo "failed: $*" >&2
        exit 1
    fi
}

train_all() {
    local failed=()
    for architecture in "${ARCHITECTURES[@]}"; do
        train "$architecture" || failed+=("$architecture")
    done
    report "${#ARCHITECTURES[@]}" architectures ${failed[@]+"${failed[@]}"}
}

train_augmentations() {
    local failed=()
    for augmentation in "${AUGMENTATIONS[@]}"; do
        train "$1" "$augmentation" || failed+=("$augmentation")
    done
    report "${#AUGMENTATIONS[@]}" augmentations ${failed[@]+"${failed[@]}"}
}

case "${1:-}" in
    prepare)
        prepare
        ;;
    all)
        require_normalization
        require_cuda
        train_all
        ;;
    augmentation)
        [[ " ${ARCHITECTURES[*]} " == *" ${2:-} "* && -n "${2:-}" ]] || usage
        [[ -z "${3:-}" || " ${AUGMENTATIONS[*]} " == *" ${3:-} "* ]] || usage
        require_normalization
        require_cuda
        if [[ -n "${3:-}" ]]; then
            train "$2" "$3"
        else
            train_augmentations "$2"
        fi
        ;;
    *)
        [[ " ${ARCHITECTURES[*]} " == *" ${1:-} "* && -n "${1:-}" ]] || usage
        require_normalization
        require_cuda
        train "$1"
        ;;
esac