# file: helpers/semantic_correlation.py
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional

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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.")
_sb = create_client(SUPABASE_URL, SUPABASE_KEY)

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


# ============================================================================
# Data access
# ============================================================================
def _get_tweet(tweet_id: str) -> dict:
    res = _sb.table("tweets").select("id,text,embedding").eq("id", tweet_id).limit(1).execute()
    if not res.data:
        raise ValueError(f"Tweet {tweet_id} not found")
    return res.data[0]

def _rpc_match_markets_v2(query_embedding: List[float], k: int = 50) -> List[dict]:
    res = _sb.rpc("match_markets_v2", {"query_embedding": query_embedding, "match_count": k}).execute()
    return res.data or []

def _llm_refine(tweet_text: str, candidates: List[Dict[str, Any]]) -> Optional[ValidatedCorrelations]:
    if not _llm_chain:
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
    return raw  # ValidatedCorrelations

# ============================================================================
# Public API
# ============================================================================
def correlate_tweet_id(
    tweet_id: str,
    top_n: int = 50,
    relevance_threshold: float = float(os.getenv("CORR_RELEVANCE_THRESHOLD", "0.6")),
) -> Dict[str, Any]:
    """
    Correlate a tweet (by id) to markets using semantic search + LLM scoring.

    Returns:
    {
      "tweet_id": str,
      "tweet_text": str,
      "timing_ms": {"vector_rpc": int, "llm": int},
      "candidates": [ {id, question, ...}, ... ],   # raw top-N vector results
      "correlations": [
          {
            "id": str,
            "question": str,
            "relevance_score": float,
            "urgency_score": float,
            "relevance_score_reasoning": str,
            "urgency_score_reasoning": str
          },
          ...
      ]
    }
    """
    tweet = _get_tweet(tweet_id)
    text = (tweet.get("text") or "").strip()
    emb = tweet.get("embedding")

    if not emb or len(emb) < 8:
        raise ValueError(f"Tweet {tweet_id} missing or invalid embedding.")

    # Stage 1: vector retrieval
    t0 = time.perf_counter()
    candidates = _rpc_match_markets_v2(emb, top_n)
    t1 = time.perf_counter()

    # Early exit if no candidates
    if not candidates:
        return {
            "tweet_id": tweet_id,
            "tweet_text": text,
            "timing_ms": {"vector_rpc": int((t1 - t0) * 1000), "llm": 0},
            "candidates": [],
            "correlations": [],
        }

    # Stage 2: LLM scoring (optional)
    t2 = time.perf_counter()
    parsed = _llm_refine(text, candidates)
    t3 = time.perf_counter()

    llm_ms = int((t3 - t2) * 1000)
    rpc_ms = int((t1 - t0) * 1000)

    correlations: List[Dict[str, Any]] = []
    if parsed and parsed.correlations:
        # filter & sort
        kept = [
            c for c in parsed.correlations
            if (c.relevance_score is not None and c.relevance_score >= relevance_threshold)
        ]
        kept.sort(key=lambda x: (-x.relevance_score, -x.urgency_score))
        correlations = [
            {
                "id": c.market.id,
                "question": c.market.question,
                "relevance_score": c.relevance_score,
                "urgency_score": c.urgency_score,
                "relevance_score_reasoning": c.relevance_score_reasoning,
                "urgency_score_reasoning": c.urgency_score_reasoning,
            }
            for c in kept
        ]

    return {
        "tweet_id": tweet_id,
        "tweet_text": text,
        "timing_ms": {"vector_rpc": rpc_ms, "llm": llm_ms},
        "candidates": candidates,
        "correlations": correlations,
    }