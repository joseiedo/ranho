#!/usr/bin/env bash
set -euo pipefail

IMAGE="ghcr.io/joseiedo/rinha-de-backend-2026"

# Get latest semver tag, default to v0.0.0
LATEST=$(git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)
LATEST=${LATEST:-v0.0.0}

# Increment patch
IFS='.' read -r MAJOR MINOR PATCH <<< "${LATEST#v}"
NEXT="v${MAJOR}.${MINOR}.$((PATCH + 1))"

echo "Current: ${LATEST} → Next: ${NEXT}"

docker buildx build --platform linux/amd64 \
  --build-arg API_RUSTFLAGS="-C target-cpu=haswell" \
  -t "${IMAGE}:${NEXT}" \
  -t "${IMAGE}:latest" \
  --push .

git tag "${NEXT}"
git push origin "${NEXT}"

echo "Done: ${IMAGE}:${NEXT}"
