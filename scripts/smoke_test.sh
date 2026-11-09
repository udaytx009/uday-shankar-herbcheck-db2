#!/usr/bin/env bash
# Usage: scripts/smoke_test.sh [base_url]   (default http://localhost:8000)
set -u
B="${1:-http://localhost:8000}"
echo "== health";  curl -s "$B/api/health"; echo
echo "== stats";   curl -s "$B/api/stats"; echo
echo "== warfarin + ginkgo"
curl -s -X POST "$B/api/check" -H 'Content-Type: application/json' -d '{"medications":["warfarin"],"herbs":["ginkgo"]}' | head -c 700; echo
echo "== typo (tumeric) -> suggestion, needs_review"
curl -s -X POST "$B/api/check" -H 'Content-Type: application/json' -d '{"medications":["warfarin"],"herbs":["tumeric"]}' | head -c 500; echo
