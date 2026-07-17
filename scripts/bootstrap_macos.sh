#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv_dir=${SCION_VENV_DIR:-"$repo_dir/.venv-gemma"}

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "Scion training requires Apple Silicon macOS" >&2
  exit 1
fi

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}

if [ ! -x "$venv_dir/bin/python" ]; then
  uv venv "$venv_dir" --python 3.13
fi

uv pip install --python "$venv_dir/bin/python" -e "$repo_dir[train,dev]"
"$venv_dir/bin/python" "$repo_dir/scripts/patch_mlx_lm_transformers5.py"
"$venv_dir/bin/python" -m mlx_vlm.lora --help >/dev/null
echo "Scion environment ready: $venv_dir"
