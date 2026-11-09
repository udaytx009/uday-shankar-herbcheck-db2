cd /workspaces/uday-shankar-herbcheck-db2 && mkdir -p docs && cat > docs/REBUILD.md <<'EOF'
# Rebuild notes (fast restart)

## What this is
HerbCheck on IBM Db2 12.1 (Community Edition, Docker) + LangChain + FastAPI.
Db2 stores the reference data, VECTOR(384) columns, and a LangChain DB2VS evidence table.
Research demo, synthetic reference data, not medical advice.

## Rebuild in about 10 minutes (new Codespace from this repo)
1. bash scripts/setup_env.sh          # makes .env with a random Db2 password (never commit or paste it)
2. set -a; source .env; set +a
3. docker run -d --name db2 --privileged -p 50000:50000 \
     -e LICENSE=accept -e DB2INSTANCE=db2inst1 \
     -e DB2INST1_PASSWORD="$DB2INST1_PASSWORD" -e DBNAME=herbdb \
     -v db2data:/database icr.io/db2_community/db2
4. docker logs -f db2                 # wait for "Setup has completed" (5-10 min first time)
5. python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
6. python -m app.ingest --reset       # schema + data + embeddings
7. uvicorn app.api:app --host 0.0.0.0 --port 8000
   (open the forwarded port 8000)
8. pytest -q                          # unit tests (no Db2 needed)

## Reuse for other projects
- app/repo.py: Db2 access, VECTOR_DISTANCE nearest-neighbour, DB2VS retrieval
- app/embeddings.py: fastembed (bge-small, 384-d) with hashing fallback
- app/chain.py: LangChain Runnable pipeline with a confidence gate and audit log
- Swap data/interaction_kb.json and sql/schema.sql for a new domain.

## Gotchas
- DB2VS applies filters after top-k, so retrieval over-fetches (k=50).
- Unknown names are only suggested, never auto-accepted.
- No record never means safe.
- Db2 Free SaaS needed a card; Docker in Codespaces did not.
- Stopped Codespaces auto-delete after the retention period; the code lives on GitHub.
EOF
git add docs/REBUILD.md && git commit -m "Add rebuild notes" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git push
