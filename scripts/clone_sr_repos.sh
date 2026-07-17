#!/usr/bin/env bash
set -e
mkdir -p external
cd external

clone_if_missing () {
  local dir="$1"
  local url="$2"
  if [ -d "$dir/.git" ]; then
    echo "[SKIP] $dir already exists"
  else
    echo "[CLONE] $url -> $dir"
    git clone "$url" "$dir"
  fi
}

clone_if_missing SPAN https://github.com/hongyuanyu/SPAN.git
clone_if_missing SAFMN https://github.com/sunny2109/SAFMN.git
clone_if_missing RFDN https://github.com/njulj/RFDN.git
clone_if_missing EFDN https://github.com/icandle/EFDN.git
clone_if_missing ASID https://github.com/saturnian77/ASID.git
clone_if_missing CATANet https://github.com/EquationWalker/CATANet.git
clone_if_missing LKDN https://github.com/stella-von/LKDN.git
clone_if_missing seemoredetails https://github.com/eduardzamfir/seemoredetails.git

cd ..
bash scripts/install_sr_adapters.sh

echo "Done. Lưu ý: mỗi repo có requirements/checkpoint riêng, cần cài theo README của repo đó."
