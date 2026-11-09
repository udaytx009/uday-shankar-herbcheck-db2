"""FastAPI service: POST /api/check, GET /api/health, GET /api/stats, and the demo UI at /."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .chain import build_chain, make_explainer
from .embeddings import get_embeddings
from .repo import Db2Repo

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Uday Shankar HerbCheck on IBM Db2", version="1.0.0")
_state: dict = {}


def get_chain():
    if "chain" not in _state:
        emb = get_embeddings()
        repo = Db2Repo(embeddings=emb)
        _state.update(repo=repo, emb=emb, chain=build_chain(repo, emb, make_explainer()))
    return _state["chain"]


def _validate(body: dict) -> dict:
    out = {}
    for key, limit in (("medications", config.MAX_MEDICATIONS), ("herbs", config.MAX_HERBS)):
        items = body.get(key)
        if not isinstance(items, list) or not items:
            raise HTTPException(422, f"'{key}' must be a non-empty list")
        if len(items) > limit:
            raise HTTPException(422, f"at most {limit} {key} per request")
        for it in items:
            if not isinstance(it, str) or not it.strip() or len(it) > config.MAX_NAME_LEN:
                raise HTTPException(422, f"each entry in '{key}' must be text of 1-{config.MAX_NAME_LEN} characters")
        out[key] = items
    return out


@app.post("/api/check")
async def check(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON object expected")
    payload = _validate(body)
    try:
        chain = get_chain()
        return JSONResponse(chain.invoke(payload))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Db2 or pipeline error: {exc.__class__.__name__}: {str(exc)[:200]}")


@app.get("/api/health")
def health():
    try:
        get_chain()
        return {"status": "ok", "db2_version": _state["repo"].version(), "embeddings": getattr(_state["emb"], "name", "custom")}
    except Exception as exc:
        raise HTTPException(503, f"{exc.__class__.__name__}: {str(exc)[:200]}")


@app.get("/api/stats")
def stats():
    try:
        get_chain()
        return _state["repo"].stats()
    except Exception as exc:
        raise HTTPException(503, f"{exc.__class__.__name__}: {str(exc)[:200]}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
