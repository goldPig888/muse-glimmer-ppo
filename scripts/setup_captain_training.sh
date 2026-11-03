#!/usr/bin/env bash
# One-time simulator, sandbox and perception install, isolated from Glimmer's env.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONDA_BIN=${CONDA_EXE:-conda}
H=$PROJECT_ROOT/data/captain/upstream/harness
if ! command -v bwrap >/dev/null; then
  echo 'Missing bubblewrap. Install it first: sudo apt-get install bubblewrap' >&2
  exit 1
fi
cd "$PROJECT_ROOT"
python3 scripts/captain_baseline.py prepare
cd "$H"
echo '[setup] Fetching pinned LIBERO and SAM service sources'
bash scripts/setup_third_party.sh --capx
create_env() {
  if [ ! -x "$1/bin/python" ]; then
    "$CONDA_BIN" create --yes --prefix "$1" "python=$2" pip
  fi
}
echo '[setup] Installing simulator environment (Python 3.10)'
create_env "$H/.venv-libero" 3.10
"$H/.venv-libero/bin/python" -m pip install -r requirements-libero.txt
"$H/.venv-libero/bin/python" -m pip install -e third_party/libero --no-deps
"$H/.venv-libero/bin/python" -m pip install -e . --no-deps
echo '[setup] Installing the persistent Python interpreter'
create_env "$H/.venv-student" 3.11.16
"$H/.venv-student/bin/python" -m pip install -r requirements-student.txt
echo '[setup] Installing separate SAM environment (no simulator extras)'
create_env "$H/.venv-tools" 3.11.16
"$H/.venv-tools/bin/python" -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
"$H/.venv-tools/bin/python" -m pip install -e "$H/third_party/cap-x/capx/third_party/sam3[notebooks]" tyro uvicorn fastapi omegaconf hydra-core
"$H/.venv-tools/bin/python" -m pip install -e "$H/third_party/cap-x" --no-deps
echo '[setup] Installing separate Molmo/vLLM environment'
create_env "$H/.venv-vllm" 3.11.16
"$H/.venv-vllm/bin/python" -m pip install 'vllm==0.29.0' 'transformers==5.17.0'
export HF_CLI="$H/.venv-vllm/bin/hf"
export CAPTAIN_MODELS_DIR=${CAPTAIN_MODELS_DIR:-$PROJECT_ROOT/data/captain/models}
echo '[setup] Downloading SAM and Molmo. Hugging Face SAM3 access is required.'
bash scripts/download_models.sh
ENV_FILE=$PROJECT_ROOT/data/captain/training.env
{
  printf 'export CAPTAIN_PYTHON=%q\n' "$H/.venv-libero/bin/python"
  printf 'export STUDENT_PYTHON=%q\n' "$H/.venv-student/bin/python"
  printf 'export TOOLS_PYTHON=%q\n' "$H/.venv-tools/bin/python"
  printf 'export VLLM_BIN=%q\n' "$H/.venv-vllm/bin/vllm"
  printf 'export SAM3_CHECKPOINT=%q\n' "$CAPTAIN_MODELS_DIR/sam3/sam3.pt"
  printf 'export MOLMO_MODEL=%q\n' "$CAPTAIN_MODELS_DIR/Molmo2-8B"
} > "$ENV_FILE"
"$H/.venv-tools/bin/python" -m pip freeze > "$PROJECT_ROOT/data/captain/sam_packages.txt"
"$H/.venv-vllm/bin/python" -m pip freeze > "$PROJECT_ROOT/data/captain/vllm_packages.txt"
echo "[setup] Installed. Configuration: $ENV_FILE"
