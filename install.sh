#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HOME/.local/share/foliolense"
BIN="$HOME/.local/bin"

echo
echo "FolioLense"
echo "========================="
echo "Installs locally in: $APP"
echo "Setup downloads Python packages and may take several minutes."
echo

command -v python3 >/dev/null 2>&1 || {
  echo "Python 3 is required."
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "NVIDIA driver not found (nvidia-smi missing)."
  exit 1
}

mkdir -p "$APP" "$BIN"

cp "$HERE/foliolense_crawl.py" "$APP/foliolense_crawl.py"
cp "$HERE/foliolense_batch.py" "$APP/foliolense_batch.py"
cp "$HERE/foliolense.sh" "$APP/foliolense.sh"
chmod +x "$APP/foliolense.sh"

echo "[1/4] Creating Python environment..."
python3 -m venv "$APP/.venv"

PY="$APP/.venv/bin/python"
PIP="$APP/.venv/bin/pip"

echo "[2/4] Updating pip..."
"$PY" -m pip install --upgrade pip

echo "[3/4] Installing PyTorch + document tools..."
"$PIP" install --upgrade torch torchvision
"$PIP" install --upgrade "sentence-transformers[image]" pymupdf pillow numpy

echo "[4/4] Checking NVIDIA GPU..."
"$PY" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "\nPyTorch installed, but CUDA is not available.\n"
        "Do not start indexing yet. Check the NVIDIA driver/PyTorch install."
    )
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
PY

ln -sf "$APP/foliolense.sh" "$BIN/foliolense"

echo
echo "Installed."
echo
echo "Launch with:"
echo "  $APP/foliolense.sh"
echo
echo "or simply:"
echo "  foliolense"
echo
echo "If 'foliolense' is not found, use the full launch path above."
echo "Choose 1 in the menu to add your first documents."
echo
