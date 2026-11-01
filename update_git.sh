#!/usr/bin/env bash
local_dir="$(pwd)"

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"commit message\""
    exit 1
fi

cd /home/peter/AI_coding/directio.net
COMMIT_MESSAGE="$1"

echo "=== Git Status ==="
git status

echo
echo "=== Adding Files ==="
git add .

echo
echo "=== Committing ==="

# Don't fail if there is nothing to commit
if git diff --cached --quiet; then
    echo "Nothing to commit."
else
    git commit -m "$COMMIT_MESSAGE"
fi

echo
echo "=== Pushing ==="
git push

echo
echo "✅ Push complete."
cd "$local_dir"
