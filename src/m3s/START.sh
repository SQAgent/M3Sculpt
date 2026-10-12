#!/usr/bin/env bash
# Start M3Sculptor: the UI and workflow share a single process; no separate backend.
set -euo pipefail
cd "$(dirname "$0")/../.."
CONDA_ENV="${M3SCULPTOR_CONDA_ENV:-m3s}"
PYTHON="$(conda info --base)/envs/$CONDA_ENV/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Cannot find Python in conda environment $CONDA_ENV; create or activate the environment first." >&2
  exit 1
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" src/m3s/app/server.py "$@"
