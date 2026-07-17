#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADAPTERS="$ROOT/external/adapters"

if [ ! -d "$ADAPTERS" ]; then
  echo "Missing adapters folder: $ADAPTERS"
  exit 1
fi

install_repo_adapters () {
  local repo="$1"
  local src="$ADAPTERS/$repo"
  local dst="$ROOT/external/$repo"

  if [ ! -d "$src" ]; then
    return 0
  fi
  if [ ! -d "$dst" ]; then
    echo "[SKIP] $repo: clone repo first (bash scripts/clone_sr_repos.sh)"
    return 0
  fi

  echo "[INSTALL] $repo adapters -> $dst"
  cp -f "$src"/* "$dst"/
}

for repo_dir in "$ADAPTERS"/*; do
  [ -d "$repo_dir" ] || continue
  install_repo_adapters "$(basename "$repo_dir")"
done

echo "Done. Adapters copied into external/* repos."
