#!/usr/bin/env bash
# Push this repository to GitHub without ever putting the token in chat or a commit.
#
#   1. Create an empty repo on github.com (no README, no .gitignore).
#   2. Put a fine-grained token with "Contents: read and write" in .github-token
#      in this folder. It is gitignored.
#   3. Run:  GITHUB_REPO=yourname/your-repo bash push.sh
#
# The token is read from the file, used once for the remote URL, and never printed.
set -euo pipefail

REPO="${GITHUB_REPO:-}"
if [ -z "$REPO" ]; then
  printf 'Set the target repo, e.g. GITHUB_REPO=utkarsh/neet-grounding bash push.sh\n'
  exit 1
fi

TOKEN="${GITHUB_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -f .github-token ]; then
  TOKEN="$(tr -d '\r\n' < .github-token)"
fi
if [ -z "$TOKEN" ]; then
  printf 'No token. Either export GITHUB_TOKEN or put one in .github-token\n'
  printf '(fine-grained, Contents: read and write, scoped to the one repo)\n'
  exit 1
fi

cd "$(dirname "$0")"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://x-access-token:${TOKEN}@github.com/${REPO}.git"
git push -u origin main
# scrub the token back out of .git/config
git remote set-url origin "https://github.com/${REPO}.git"
printf '\nPushed. Remote reset to the credential-free URL: https://github.com/%s\n' "$REPO"
