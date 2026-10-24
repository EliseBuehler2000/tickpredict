#!/bin/sh
# Copy a trained release (models/<version>/, including the untracked best.pt
# weights) from the training machine to this Mac, then check every checksum
# recorded in its manifest.
#
#   scripts/pull_release.sh v2.0 [ssh-host]
set -eu
version=${1:?usage: scripts/pull_release.sh VERSION [ssh-host]}
host=${2:-spark-4ec2-tailscale}
root=$(cd "$(dirname "$0")/.." && pwd)
[ -e "$root/models/$version/manifest.json" ] && { echo "models/$version already exists; not overwriting." >&2; exit 1; }
ssh "$host" "cd ~/tickpredict && test -f models/$version/manifest.json && tar -cf - models/$version" | tar -xf - -C "$root"
cd "$root" && .venv/bin/python - "$version" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest = json.loads(Path(f"models/{sys.argv[1]}/manifest.json").read_text())
bad = [f["release_path"] for f in manifest["files"]
       if hashlib.sha256(Path(f["release_path"]).read_bytes()).hexdigest() != f["sha256"]]
print(f"{len(manifest['files']) - len(bad)}/{len(manifest['files'])} files match the manifest")
sys.exit(1 if bad else 0)
PY
