# helpers/correlation_engine.py
from __future__ import annotations

import os, json, time, math
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage

from helpers.embeddings import generate_embeddings
from helpers.supabase_db import (
    get_unprocessed_tweets_batch,   # -> [{ id, text, embedding, ... }]
    upsert_tweet_embeddings,        # -> set embedding by id
    rpc_match_markets,              # -> top-K market rows with id, question, embedding_text...
    upsert_correlations,            # -> into tweet_market_correlations
    mark_tweets_processed,          # -> mark processed true (accepts list)
)

load_dotenv()

# --- Category catalog (canonical) ---
_CATEGORIES = [
    {"id": "268ad0cd-dd58-4cd2-b91b-d48497e6021d", "name": "Financial"},
    {"id": "6695e38d-c013-46ec-8bbb-7ff81e1d515a", "name": "Politics"},
    {"id": "7cf3f17f-ee49-44fa-a202-45224d6bcede", "name": "Sports"},
    {"id": "a3d7d328-90b1-4a67-a631-d8138da3890d", "name": "Crypto"},
    {"id": "bb952fa4-9316-489f-a992-3ea10b025284", "name": "Culture"},
    {"id": "dd8fc4ce-47d0-4497-84b5-6c5b34e54bf1", "name": "Content Creators"},
    {"id": "e4d5e2fe-98ab-4a50-9e70-cf8178735d75", "name": "Mention Markets"},
    {"id": "fd9f393d-44de-4c90-8e50-fc447d10cf07", "name": "Science"},
]
_CATEGORY_NAME_TO_ID = {c["name"].lower(): c["id"] for c in _CATEGORIES}
_CANONICAL_NAMES = {c["name"] for c in _CATEGORIES}

# lightweight synonym map so the LLM can be a little sloppy
_CATEGORY_SYNONYMS = {
    "finance": "Financial",
    "economic": "Financial",
    "economy": "Financial",

    "political": "Politics",
    "gov": "Politics",
    "government": "Politics",
    "policy": "Politics",

    "sport": "Sports",
    "football": "Sports",
    "basketball": "Sports",

    "creator": "Content Creators",
    "creators": "Content Creators",
    "influencers": "Content Creators",

    "mention": "Mention Markets",
    "mentions": "Mention Markets",

    "sci": "Science",
    "research": "Science",
}

def _normalize_category_name(s: str | None) -> str | None:
    if not s:
        return None
    key = s.strip().lower()
    # exact canonical
    for name in _CANONICAL_NAMES:
        if key == name.lower():
            return name
    # synonyms
    if key in _CATEGORY_SYNONYMS:
        return _CATEGORY_SYNONYMS[key]
    return None

# ---------------------------
# Config knobs (env-driven)
# ---------------------------
TOP_N_CANDIDATES        = int(os.getenv("CORR_TOP_K", "30"))
RELEVANCE_THRESHOLD     = float(os.getenv("CORR_RELEVANCE_THRESHOLD", "0.6"))
MAX_TWEETS_PER_CYCLE    = int(os.getenv("CORR_MAX_TWEETS", "500"))
ENGINE_VERSION          = os.getenv("ENGINE_VERSION", "v1.0")
# LLM_MODEL_TAG           = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash-lite-preview-06-17")
LLM_MODEL_TAG           = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-lite")
CORR_MAX_WORKERS        = int(os.getenv("CORR_MAX_WORKERS", "4"))          # thread pool size
LLM_MAX_CONCURRENCY     = int(os.getenv("LLM_MAX_CONCURRENCY", "2"))       # cap LLM parallelism

# ---------------------------
# Pydantic schema (unchanged)
# ---------------------------
class LLMMarketResponse(BaseModel):
    id: str = Field(description="The unique ID of the market.")
    question: str = Field(description="The question of the market.")


class ScoredCorrelation(BaseModel):
    market: LLMMarketResponse
    relevance_score: float = Field(description="0.0 to 1.0")
    relevance_score_reasoning: str
    urgency_score: float = Field(description="0.0 to 1.0")
    urgency_score_reasoning: str

    # NEW (optional so old outputs still parse)
    category_name: Optional[str] = Field(
        default=None,
        description="One of: Financial, Politics, Sports, Crypto, Culture, Content Creators, Mention Markets, Science"
    )
    category_reasoning: Optional[str] = None

class ValidatedCorrelations(BaseModel):
    correlations: List[ScoredCorrelation]

# ---------------------------
# Chat model + prompt (same)
# ---------------------------
_model = init_chat_model(
    LLM_MODEL_TAG,
    model_provider="google_genai",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
)
_structured_model = _model.with_structured_output(ValidatedCorrelations)

_prompt = ChatPromptTemplate.from_template(
    """
You are an expert financial and geopolitical analyst. Analyze a breaking news tweet and evaluate its connection to a list of potential prediction markets.

Return only markets with relevance_score > 0.1. Score both relevance and urgency (0–1). Include brief reasoning for each, citing the exact tweet phrase that supports inclusion.

Also, for each kept market, assign a single `category_name` using EXACTLY one of these labels:
Financial, Politics, Sports, Crypto, Culture, Content Creators, Mention Markets, Science
If unsure, choose the closest. Include a short `category_reasoning`.

<TWEET>
{tweet_json}
</TWEET>

<CANDIDATE_MARKETS>
{markets_json}
</CANDIDATE_MARKETS>
""".strip()
)
_chain = _prompt | _structured_model

# ---------------------------
# LLM helpers (robust parse)
# ---------------------------
import threading
_LLM_GATE = threading.Semaphore(LLM_MAX_CONCURRENCY)

def _extract_json_from_message(msg: BaseMessage) -> Optional[str]:
    try:
        fc = msg.additional_kwargs.get("function_call")
        if fc and isinstance(fc, dict) and "arguments" in fc:
            return fc["arguments"]
    except Exception:
        pass
    try:
        tcs = msg.additional_kwargs.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            fn = tcs[0].get("function", {})
            if "arguments" in fn:
                return fn["arguments"]
    except Exception:
        pass
    return msg.content

def _parse_llm_output(raw: Any) -> Optional[ValidatedCorrelations]:
    try:
        if isinstance(raw, ValidatedCorrelations):
            return raw
        if hasattr(raw, "model_dump"):
            return ValidatedCorrelations(**raw.model_dump())
        if isinstance(raw, dict):
            return ValidatedCorrelations(**raw)
        text: Optional[str] = None
        if isinstance(raw, BaseMessage):
            text = _extract_json_from_message(raw)
        elif isinstance(raw, str):
            text = raw
        elif hasattr(raw, "content"):
            text = getattr(raw, "content")
        if text:
            s = text.strip()
            if s.startswith("```"):
                s = s.strip("`")
                if "\n" in s: s = s.split("\n", 1)[1]
            return ValidatedCorrelations.model_validate_json(s)
    except Exception:
        pass
    try:
        preview = (raw.content if isinstance(raw, BaseMessage) else str(raw))[:400]
        print(f'{{"type":"warn","msg":"unable_to_parse_llm","preview":{json.dumps(preview)}}}')
    except Exception:
        print('{"type":"warn","msg":"unable_to_parse_llm_no_preview"}')
    return None

def _llm_invoke(payload: dict, retries: int = 1, backoff_s: float = 1.5):
    for attempt in range(retries + 1):
        try:
            with _LLM_GATE:
                return _chain.invoke(payload)
        except Exception as e:
            if attempt >= retries:
                raise
            time.sleep(backoff_s)

# ---------------------------
# Embedding support
# ---------------------------
def _ensure_tweet_embeddings(tweets: List[Dict[str, Any]]):
    todo = [(t["id"], (t.get("text") or "").replace("\n", " ").strip())
            for t in tweets if not t.get("embedding")]
    todo = [(tid, txt) for (tid, txt) in todo if txt]
    if not todo:
        return
    ids = [tid for tid,_ in todo]
    texts = [txt for _,txt in todo]
    embs = generate_embeddings(texts)
    id_to_emb = {}
    for (tid,_), emb in zip(todo, embs):
        id_to_emb[tid] = (emb.tolist() if hasattr(emb, "tolist") else emb)
    upsert_tweet_embeddings(id_to_emb)

# ---------------------------
# Per-tweet worker
# ---------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _process_one_tweet(tweet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a compact record with timings and (stored) correlation count.
    """
    tid   = tweet["id"]
    ttext = (tweet.get("text") or "").strip()
    temb  = tweet.get("embedding")

    started_at = _now_iso()
    t0_wall = time.perf_counter()

    if not temb or (hasattr(temb, "__len__") and len(temb) < 8):
        mark_tweets_processed([tid])
        return {
            "tweet_id": tid, "ok": False, "error": "missing_or_short_embedding",
            "started_at": started_at, "finished_at": _now_iso(), "wall_ms": int((time.perf_counter()-t0_wall)*1000)
        }

    # Stage 1: vector match
    try:
        t_vec0 = time.perf_counter()
        candidates = rpc_match_markets(temb, TOP_N_CANDIDATES) or []
        vec_ms = int((time.perf_counter() - t_vec0) * 1000)
        candidates = [c for c in candidates if c.get("id")]
        if not candidates:
            mark_tweets_processed([tid])
            return {
                "tweet_id": tid, "ok": True, "stored": 0,
                "timing_ms": {"vector_rpc": vec_ms, "llm": 0},
                "started_at": started_at, "finished_at": _now_iso(),
                "wall_ms": int((time.perf_counter()-t0_wall)*1000),
            }
    except Exception as e:
        mark_tweets_processed([tid])
        return {
            "tweet_id": tid, "ok": False, "error": f"vector_rpc_error: {e}",
            "started_at": started_at, "finished_at": _now_iso(), "wall_ms": int((time.perf_counter()-t0_wall)*1000)
        }

    markets_for_llm = [{"id": c.get("id"), "question": c.get("question"), "embedding_text": c.get("embedding_text")} for c in candidates]

    # Stage 2: LLM refine
    try:
        t_llm0 = time.perf_counter()
        raw = _llm_invoke({
            "markets_json": json.dumps(markets_for_llm, ensure_ascii=False, separators=(",", ":")),
            "tweet_json": json.dumps({"text": ttext}, ensure_ascii=False, separators=(",", ":")),
        }, retries=1)
        llm_ms = int((time.perf_counter() - t_llm0) * 1000)
    except Exception as e:
        mark_tweets_processed([tid])
        return {
            "tweet_id": tid, "ok": False, "error": f"llm_error: {e}",
            "started_at": started_at, "finished_at": _now_iso(), "wall_ms": int((time.perf_counter()-t0_wall)*1000)
        }

    parsed = _parse_llm_output(raw)
    if not parsed or not parsed.correlations:
        mark_tweets_processed([tid])
        return {
            "tweet_id": tid, "ok": True, "stored": 0,
            "timing_ms": {"vector_rpc": vec_ms, "llm": llm_ms},
            "started_at": started_at, "finished_at": _now_iso(),
            "wall_ms": int((time.perf_counter()-t0_wall)*1000),
        }

    # rows = []
    # for corr in parsed.correlations:
    #     try:
    #         if corr.relevance_score < RELEVANCE_THRESHOLD:
    #             continue
    #         rows.append({
    #             "tweet_id": tid,
    #             "market_id": corr.market.id,
    #             "relevance_score": corr.relevance_score,
    #             "relevance_reason": corr.relevance_score_reasoning,
    #             "urgency_score": corr.urgency_score,
    #             "urgency_reason": corr.urgency_score_reasoning,
    #             "engine_version": ENGINE_VERSION,
    #             "llm_model": LLM_MODEL_TAG,
    #             "tokens_input": None,
    #             "tokens_output": None,
    #             "cost_usd": None,
    #             "metadata": {"latency_ms": llm_ms},
    #         })
    #     except Exception:
    #         # skip malformed record
    #         continue
    rows = []
    for corr in parsed.correlations:
        try:
            if corr.relevance_score < RELEVANCE_THRESHOLD:
                continue

            # NEW: normalize + map to UUID
            raw_cat = getattr(corr, "category_name", None)
            cat_name = _normalize_category_name(raw_cat)
            cat_id = _CATEGORY_NAME_TO_ID.get(cat_name.lower()) if cat_name else None

            meta = {"latency_ms": llm_ms}
            if cat_id:
                meta["category_id"] = cat_id  # stash UUID in metadata without schema changes

            rows.append({
                "tweet_id": tid,
                "market_id": corr.market.id,
                "relevance_score": corr.relevance_score,
                "relevance_reason": corr.relevance_score_reasoning,
                "urgency_score": corr.urgency_score,
                "urgency_reason": corr.urgency_score_reasoning,
                "engine_version": ENGINE_VERSION,
                "llm_model": LLM_MODEL_TAG,
                "tokens_input": None,
                "tokens_output": None,
                "cost_usd": None,
                "metadata": meta,

                # NEW: this column already exists in your sample (was null); now we fill it.
                "llm_category_name": cat_name,
            })
        except Exception as e:
            print(f"  !! skipping malformed correlation for tweet {tid}: {e}")

    stored = 0
    if rows:
        try:
            upsert_correlations(rows)
            stored = len(rows)
        except Exception as e:
            # store failed, but still mark processed
            pass

    mark_tweets_processed([tid])

    return {
        "tweet_id": tid,
        "ok": True,
        "stored": stored,
        "timing_ms": {"vector_rpc": vec_ms, "llm": llm_ms},
        "started_at": started_at,
        "finished_at": _now_iso(),
        "wall_ms": int((time.perf_counter()-t0_wall)*1000),
    }

# ---------------------------
# Public: parallel driver
# ---------------------------
def run_correlation_engine_parallel(stream_ndjson: bool = True) -> int:
    """
    Fetch batch, ensure embeddings, then process tweets in parallel.
    If stream_ndjson=True, prints one NDJSON line per tweet as it finishes.
    Returns number of tweets processed.
    """
    unprocessed = get_unprocessed_tweets_batch(limit=MAX_TWEETS_PER_CYCLE) or []
    if not unprocessed:
        if stream_ndjson:
            print(json.dumps({"type":"batch","status":"empty","ts":_now_iso()}))
        return 0

    # Make sure all tweets have embeddings first (vector calls are fast once ready)
    _ensure_tweet_embeddings(unprocessed)

    started = time.perf_counter()
    batch_meta = {
        "type": "batch_start",
        "ts": _now_iso(),
        "n": len(unprocessed),
        "top_n": TOP_N_CANDIDATES,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "max_workers": CORR_MAX_WORKERS,
        "llm_max_concurrency": LLM_MAX_CONCURRENCY,
    }
    if stream_ndjson: print(json.dumps(batch_meta))

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=CORR_MAX_WORKERS) as ex:
        futs = {ex.submit(_process_one_tweet, tw): tw["id"] for tw in unprocessed}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            if stream_ndjson:
                print(json.dumps({"type":"tweet_done", **rec}, ensure_ascii=False))

    total_ms = int((time.perf_counter() - started) * 1000)

    # small rollups
    walls = [r["wall_ms"] for r in results if r.get("wall_ms") is not None]
    vecs  = [r["timing_ms"]["vector_rpc"] for r in results if r.get("ok") and r.get("timing_ms")]
    llms  = [r["timing_ms"]["llm"] for r in results if r.get("ok") and r.get("timing_ms")]

    def _avg(xs): return int(sum(xs)/len(xs)) if xs else None
    def _p95(xs):
        if not xs: return None
        xs = sorted(xs); i = int(math.ceil(0.95 * len(xs))) - 1
        return xs[i]

    summary = {
        "type": "batch_end",
        "ts": _now_iso(),
        "processed": len(results),
        "total_ms": total_ms,
        "stored_rows": int(sum(r.get("stored", 0) for r in results if r.get("ok"))),
        "latency": {
            "wall_ms": {"avg": _avg(walls), "p95": _p95(walls), "max": max(walls) if walls else None},
            "vector_rpc_ms": {"avg": _avg(vecs), "p95": _p95(vecs)},
            "llm_ms": {"avg": _avg(llms), "p95": _p95(llms)},
        },
    }
    if stream_ndjson:
        print(json.dumps(summary))

    return len(results)