from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from supabase import create_client

# --- optional LLM deps ---
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

# ============================================================================
# Environment & Clients
# ============================================================================
load_dotenv()

# -- Supabase init ------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# -- LLM init (optional) ------------------------------------------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash-lite-preview-06-17")

# ============================================================================
# LLM Correlation Scoring (Stage 2)
# ============================================================================
class LLMMarketResponse(BaseModel):
    id: str = Field(description="The unique ID of the market.")
    question: str = Field(description="The question of the market.")

class ScoredCorrelation(BaseModel):
    market: LLMMarketResponse
    relevance_score: float = Field(description="0.0 to 1.0")
    relevance_score_reasoning: str
    urgency_score: float = Field(description="0.0 to 1.0")
    urgency_score_reasoning: str

class ValidatedCorrelations(BaseModel):
    correlations: List[ScoredCorrelation]

_llm_chain = None
if GEMINI_KEY:
    _raw_model = init_chat_model(
        LLM_MODEL_NAME,
        model_provider="google_genai",
        google_api_key=GEMINI_KEY,
        temperature=0.5,
    )
    _structured_model = _raw_model.with_structured_output(ValidatedCorrelations)
    _prompt = ChatPromptTemplate.from_template(
        """
You are an expert financial and geopolitical analyst. Analyze a breaking news tweet and evaluate its connection to a list of potential prediction markets.

Return only markets with relevance_score > 0.1. Score both relevance and urgency (0–1). Include brief reasoning for each, citing the exact tweet phrase that supports inclusion.

<TWEET>
{tweet_json}
</TWEET>

<CANDIDATE_MARKETS>
{markets_json}
</CANDIDATE_MARKETS>
        """.strip()
    )
    _llm_chain = _prompt | _structured_model

def llm_refine(tweet_text: str, candidates: List[Dict[str, Any]]) -> ValidatedCorrelations | None:
    """Call LLM to score relevance/urgency for the candidate list."""
    if not _llm_chain:
        print("(LLM not configured: set GEMINI_API_KEY to enable stage-2 refine)")
        return None
    markets_for_llm = [{
        "id": str(c.get("id")),
        "question": c.get("question"),
        "embedding_text": c.get("embedding_text"),
    } for c in candidates]
    raw = _llm_chain.invoke({
        "markets_json": json.dumps(markets_for_llm, ensure_ascii=False, separators=(",", ":")),
        "tweet_json": json.dumps({"text": tweet_text}, ensure_ascii=False, separators=(",", ":")),
    })
    return raw  # already ValidatedCorrelations

# ============================================================================
# Supabase Data Access
# ============================================================================
def get_tweet(tweet_id: str) -> dict:
    """Fetch a tweet row from supabase by id (expects id, text, embedding)."""
    res = sb.table("tweets").select("id,text,embedding").eq("id", tweet_id).limit(1).execute()
    if not res.data:
        raise ValueError(f"Tweet {tweet_id} not found")
    return res.data[0]

def rpc_match_markets_v2(query_embedding: List[float], k: int = 50) -> List[dict]:
    """Vector semantic search candidates."""
    res = sb.rpc("match_markets_v2", {"query_embedding": query_embedding, "match_count": k}).execute()
    return res.data or []

# ============================================================================
# Orchestrator
# ============================================================================
def run_pipeline(
    tweet_id: str,
    top_n: int = 50,
    relevance_threshold: float = 0.6,
):
    """Run semantic retrieval then LLM-refine for one tweet."""
    tweet = get_tweet(tweet_id)
    text = (tweet.get("text") or "").strip()
    emb = tweet.get("embedding")

    print("Tweet content:")
    print(text or "(empty)")
    print()

    if not emb or len(emb) < 8:
        print(f"Tweet {tweet_id} missing or invalid embedding.")
        return

    # Stage 1: semantic (timed)
    t0 = time.perf_counter()
    candidates = rpc_match_markets_v2(emb, top_n)
    t1 = time.perf_counter()
    rpc_ms = int((t1 - t0) * 1000)

    if not candidates:
        print("(no candidates from semantic search)")
        return

    print(f"Top {len(candidates)} vector candidates (RPC {rpc_ms} ms):")
    for i, c in enumerate(candidates, 1):
        title = c.get("question") or c.get("title") or str(c.get("id"))
        print(f"{i:2d}. {title}")
    print()

    # Stage 2: LLM refine (optional; timed)
    t2 = time.perf_counter()
    parsed = llm_refine(text, candidates)
    t3 = time.perf_counter()
    if parsed is None:
        return

    llm_ms = int((t3 - t2) * 1000)
    print(f"LLM refine latency: {llm_ms} ms\n")

    # Filter & sort LLM-scored results
    correlations = [
        c for c in (parsed.correlations or [])
        if (c.relevance_score is not None and c.relevance_score >= relevance_threshold)
    ]
    correlations.sort(key=lambda x: (-x.relevance_score, -x.urgency_score))

    if not correlations:
        print(f"No markets met relevance_threshold={relevance_threshold}.")
        return

    print(f"LLM-selected correlations (>= {relevance_threshold:.2f} relevance):")
    for i, corr in enumerate(correlations, 1):
        print(
            f"{i:2d}. [{corr.market.id}] {corr.market.question}\n"
            f"     relevance={corr.relevance_score:.2f}  urgency={corr.urgency_score:.2f}\n"
            f"     why: {corr.relevance_score_reasoning}\n"
            f"     when: {corr.urgency_score_reasoning}\n"
        )

# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Semantic (vector) → LLM pipeline for a single tweet id.")
    ap.add_argument("tweet_id", help="tweet id in your 'tweets' table")
    ap.add_argument("--top-n", type=int, default=50, help="top-K semantic candidates (default 50)")
    ap.add_argument("--relevance-threshold", type=float, default=float(os.getenv("CORR_RELEVANCE_THRESHOLD", "0.6")))
    args = ap.parse_args()

    run_pipeline(
        args.tweet_id,
        top_n=args.top_n,
        relevance_threshold=args.relevance_threshold,
    )