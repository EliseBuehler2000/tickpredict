#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'Create the local environment first; see README.md under Setup.'
  exit 1
fi
# Isolate new local Jupyter state from the archived machine-specific settings.
export JUPYTER_CONFIG_DIR="$PWD/.jupyter/config"
export JUPYTER_DATA_DIR="$PWD/.jupyter/data"
export JUPYTER_RUNTIME_DIR="$PWD/.jupyter/runtime"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
exec .venv/bin/python -m jupyterlab --no-browser \
  --ServerApp.ip=127.0.0.1 --ServerApp.port=8891 --ServerApp.port_retries=0 \
  --ServerApp.root_dir="$PWD" --ServerApp.default_url=/lab/tree/testing.ipynb
