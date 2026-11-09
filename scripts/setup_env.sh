#!/usr/bin/env bash
# Creates .env with a random Db2 password, so there is no placeholder to forget.
# Safe to re-run: it never overwrites an existing .env.
set -eu
cd "$(dirname "$0")/.."
if [ -f .env ]; then echo ".env already exists; leaving it alone."; exit 0; fi
PW="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 20)"
sed "s/__GENERATED__/$PW/g" .env.example > .env
chmod 600 .env
echo "Created .env with a random Db2 password (not committed: .env is gitignored)."
