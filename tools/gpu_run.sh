#!/bin/bash
set -euo pipefail

finish() {
    status=$?
    printf '%s\n' "$status" > run.exit.tmp
    mv run.exit.tmp run.exit
}
trap finish EXIT

export PATH="$PWD:$PATH"
export HF_HOME="$PWD/hf-cache"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1
export UV_FROZEN=1
export UV_PYTHON_DOWNLOADS=never
export PYTHONUNBUFFERED=1

mkdir -p "$HOME/.cache"
if [ ! -e "$HOME/.cache/autoresearch" ]; then
    ln -s "$PWD/autoresearch" "$HOME/.cache/autoresearch"
fi

uv run prepare.py > prepare.log 2>&1
uv run train.py > run.log 2>&1
