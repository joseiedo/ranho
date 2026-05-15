#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="./spec/test"

# docker buildx build --platform linux/amd64 -t rinha-2026:local .
#
echo "==> Bringing up containers..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d --wait

echo "==> Waiting for /ready..."
until curl -sf http://localhost:9999/ready > /dev/null; do
  sleep 1
done
echo "==> API is ready."

echo "==> Running k6 stress test..."
k6 run "$TEST_DIR/test.js"

echo "==> Results:"
jq . "./test/results.json"

echo "==> Tearing down containers..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
