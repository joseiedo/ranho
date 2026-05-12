#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$SCRIPT_DIR/../rinha-de-backend-2026/test"

echo "==> Bringing up containers..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d --wait

echo "==> Running k6 stress test..."
export K6_NO_USAGE_REPORT=true
k6 run "$TEST_DIR/test.js" > /dev/null 2>&1

echo "==> Results:"
jq . "$TEST_DIR/results.json"

echo "==> Tearing down containers..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
