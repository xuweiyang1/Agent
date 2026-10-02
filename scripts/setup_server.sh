#!/usr/bin/env bash
# One-time server setup for a machine that cannot reach GitHub.
#
# Run this from a JupyterLab terminal, or over SSH:
#
#     bash setup_server.sh
#
# It creates a bare repo that your local machine can push into over SSH.
# GitHub stays the public archive; this bare repo is the channel that
# actually reaches the server.

set -euo pipefail

REPO_NAME="${REPO_NAME:-agent}"
BARE="${BARE:-$HOME/$REPO_NAME.git}"
WORK="${WORK:-$HOME/$REPO_NAME}"

echo "==> checking prerequisites"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed." >&2
  echo "  Debian/Ubuntu : sudo apt install -y git" >&2
  echo "  conda/mamba   : conda install -y git" >&2
  exit 1
fi

PY=""
for candidate in python3 python3.12 python3.11 python3.10; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY="$candidate"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "ERROR: need python3 >= 3.10, found: $(python3 --version 2>&1 || echo none)" >&2
  echo "       Do not patch the source to run on 3.9; the type hints use" >&2
  echo "       'X | None' which is a 3.10 syntax feature." >&2
  exit 1
fi

echo "    git    $(git --version | awk '{print $3}')"
echo "    python $($PY -c 'import platform; print(platform.python_version())')"
echo "    home   $HOME"

echo "==> bare repo at $BARE"
if [ -d "$BARE" ]; then
  echo "    already exists, leaving it alone"
else
  git init --bare --initial-branch=main "$BARE"
  echo "    created"
fi

echo "==> working tree at $WORK"
if [ -d "$WORK/.git" ]; then
  echo "    already a checkout, leaving it alone"
elif [ -d "$WORK" ] && [ -n "$(ls -A "$WORK" 2>/dev/null)" ]; then
  echo "    WARNING: $WORK exists and is not empty, so it was NOT cloned."
  echo "    Move or empty it, then run:"
  echo "        git clone $BARE $WORK"
else
  git clone "$BARE" "$WORK" 2>/dev/null || echo "    bare repo is empty, nothing to clone yet"
fi

echo
echo "==> next, from your LOCAL machine"

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "  If this host is directly reachable over SSH:"
echo "      git remote add server ssh://$(whoami)@${IP:-SERVER_IP}/$BARE"
echo "      git push -u server main"
echo
echo "  If it is NOT reachable, forward its sshd to a local port instead:"
echo "      ssh -N -L 2222:localhost:22 $(whoami)@${IP:-GATEWAY_IP}"
echo "      git remote add server ssh://$(whoami)@127.0.0.1:2222/$BARE"
echo "      git push -u server main"
echo
echo "  Then verify on this host:"
echo "      cd $WORK && $PY -m unittest discover -s tests -t . -v"