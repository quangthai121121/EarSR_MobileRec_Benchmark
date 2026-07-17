#!/usr/bin/env bash
# Fix: ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
# Affects BasicSR-based repos (SPAN, SAFMN, ...) with newer torchvision (>=0.17).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
patched=0

patch_file () {
  local file="$1"
  if [ ! -f "$file" ]; then
    return 0
  fi
  if grep -q 'from torchvision.transforms.functional_tensor import rgb_to_grayscale' "$file"; then
    python3 - "$file" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
new = (
    "try:\n"
    "    from torchvision.transforms.functional import rgb_to_grayscale\n"
    "except ImportError:\n"
    "    from torchvision.transforms.functional_tensor import rgb_to_grayscale"
)
if old not in text:
    raise SystemExit(0)
path.write_text(text.replace(old, new, 1))
print(f"[PATCH] {path}")
PY
    patched=$((patched + 1))
  else
    echo "[SKIP] already patched or no match: $file"
  fi
}

for repo in SPAN SAFMN LKDN CATANet seemoredetails; do
  patch_file "$ROOT/external/$repo/basicsr/data/degradations.py"
done

# Also patch site-packages basicsr if installed separately
if command -v python >/dev/null 2>&1; then
  site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
  site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
else
  site=""
fi
if [ -n "$site" ]; then
  patch_file "$site/basicsr/data/degradations.py"
fi

echo "Done. Patched $patched file(s)."
