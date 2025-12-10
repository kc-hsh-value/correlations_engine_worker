# helpers/supabase_db.py
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from helpers.supabase_client import get_supabase
from helpers.embeddings import generate_embeddings

# ========= BASIC QUERIES =========

def get_market_count() -> int:
    """
    Returns total markets count.
    Uses RPC-ish count by selecting minimal column w/ count='exact'.
    """
    sb = get_supabase()
    # Minimal count query on primary key
    resp = sb.table("markets").select("id", count="exact").execute()
    if resp.count is None:
        return 0
    return int(resp.count)


def prune_expired_markets() -> int:
    sb = get_supabase()
    res = sb.rpc("prune_markets").execute()
    return len(res.data or [])

def add_discord_subscription(guild_id: str, channel_id: str, owner_user_id: Optional[str] = None) -> None:
    """
    Adds or updates a Discord subscription. Because we didn't create a unique constraint,
    we do an upsert-like behavior: try to find by (guild_id, channel_id), insert if missing.
    """
    sb = get_supabase()

    # Check if exists
    exists = (
        sb.table("discord_subscriptions")
        .select("id")
        .eq("guild_id", guild_id)
        .eq("channel_id", channel_id)
        .limit(1)
        .execute()
    )
    if exists.data:
        # Ensure active + maybe update owner
        sb.table("discord_subscriptions").update({
            "is_active": True,
            "owner_user_id": owner_user_id
        }).eq("id", exists.data[0]["id"]).execute()
    else:
        sb.table("discord_subscriptions").insert({
            "guild_id": guild_id,
            "channel_id": channel_id,
            "is_active": True,
            "owner_user_id": owner_user_id
        }).execute()

def add_telegram_subscription(chat_id: str, owner_user_id: Optional[str] = None) -> None:
    sb = get_supabase()

    exists = (
        sb.table("telegram_subscriptions")
        .select("id")
        .eq("chat_id", chat_id)
        .limit(1)
        .execute()
    )
    if exists.data:
        sb.table("telegram_subscriptions").update({
            "is_active": True,
            "owner_user_id": owner_user_id
        }).eq("id", exists.data[0]["id"]).execute()
    else:
        sb.table("telegram_subscriptions").insert({
            "chat_id": chat_id,
            "is_active": True,
            "owner_user_id": owner_user_id
        }).execute()

# ========= POLYMARKET FETCH =========

def get_markets(end_date_min: Optional[str] = None, start_date_min: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetches markets from Polymarket's gamma API with pagination.
    """
    all_markets: List[Dict[str, Any]] = []
    offset = 0
    PAGE_SIZE = 500

    while True:
        if start_date_min and end_date_min:
            url = f"https://gamma-api.polymarket.com/markets?start_date_min={start_date_min}&end_date_min={end_date_min}&limit={PAGE_SIZE}&offset={offset}"
        elif end_date_min and not start_date_min:
            url = f"https://gamma-api.polymarket.com/markets?end_date_min={end_date_min}&limit={PAGE_SIZE}&offset={offset}"
        elif start_date_min and not end_date_min:
            url = f"https://gamma-api.polymarket.com/markets?start_date_min={start_date_min}&limit={PAGE_SIZE}&offset={offset}"
        else:
            url = f"https://gamma-api.polymarket.com/markets?limit={PAGE_SIZE}&offset={offset}"

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        page_data = resp.json()

        if not page_data:
            break

        all_markets.extend(page_data)

        if len(page_data) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(0.4)

    return all_markets

# ========= EMBEDDINGS & INSERT =========

def _extract_prices(market: Dict[str, Any]) -> Tuple[float, float]:
    """
    outcomePrices is usually a JSON array string. Safely parse.
    Returns (yes_price, no_price)
    """
    yes_price = 0.0
    no_price = 0.0
    try:
        outcome_prices_json = market.get("outcomePrices", "[]")
        prices = json.loads(outcome_prices_json) if isinstance(outcome_prices_json, str) else outcome_prices_json
        if isinstance(prices, list):
            yes_price = float(prices[0]) if len(prices) > 0 and prices[0] is not None else 0.0
            no_price = float(prices[1]) if len(prices) > 1 and prices[1] is not None else 0.0
    except Exception:
        pass
    return yes_price, no_price

def _derive_url_and_parent(market: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
    """
    Returns (market_url, parent_event_id, embedding_text)
    """
    question = market.get("question", "") or ""
    desc = market.get("description", "") or ""
    embedding_text = f"Question: {question}\nDescription: {desc}"

    parent_event_id = None
    parent_slug = market.get("slug") or ""

    events = market.get("events")
    if isinstance(events, list) and events:
        ev = events[0] or {}
        ev_slug = ev.get("ticker")
        parent_event_id = ev.get("id")
        if ev_slug:
            parent_slug = ev_slug

    market_url = f"https://polymarket.com/event/{parent_slug}" if parent_slug else f"https://polymarket.com/market/{market.get('slug','')}"
    return market_url, parent_event_id, embedding_text

def insert_markets(markets_data, embeddings) -> int:
    if not markets_data:
        return 0
    if len(markets_data) != len(embeddings):
        raise ValueError("Number of markets and embeddings mismatch.")

    rows = []
    for market, emb in zip(markets_data, embeddings):
        yes_price, no_price = _extract_prices(market)
        market_url, parent_event_id, embedding_text = _derive_url_and_parent(market)
        rows.append({
            "id": market.get("id"),
            "question": market.get("question"),
            "slug": market.get("slug"),
            "market_url": market_url,
            "parent_event_id": parent_event_id,
            "image_url": market.get("image"),
            "yes_price": yes_price,
            "no_price": no_price,
            "end_date_utc": market.get("endDate"),
            "embedding_text": embedding_text,
            "embedding": (emb.tolist() if hasattr(emb, "tolist") else emb),
            "is_active": True,
            # ⚠️ OPTIONAL: drop or shrink this to reduce payload size
            "full_data_json": market,
        })

    sb = get_supabase()
    BATCH = 300
    total = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i+BATCH]
        print(f"[seed] upserting markets batch {i//BATCH + 1}/{(len(rows)-1)//BATCH + 1} ({len(batch)} rows)…")
        sb.table("markets").upsert(
            batch,
            on_conflict="id",
            returning="minimal",  # <- critical
        ).execute()
        total += len(batch)
    return total


    
def seed_database_if_empty(generate_embeddings_fn) -> None:
    """
    If markets table is empty, fetch all active markets, embed, and insert.
    generate_embeddings_fn: callable(List[str]) -> List[np.ndarray]
    """
    count = get_market_count()
    if count > 0:
        print(f"[seed] markets already present: {count}. Skipping seed.")
        return

    print("[seed] fetching all active markets…")
    now_utc_str = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    all_markets = get_markets(end_date_min=now_utc_str)

    if not all_markets:
        print("[seed] no markets fetched. aborting.")
        return

    texts = [
        f"Question: {m.get('question','')}\nDescription: {m.get('description','')}"
        for m in all_markets
    ]
    print(f"[seed] generating {len(texts)} embeddings…")
    embs = generate_embeddings_fn(texts)

    print("[seed] inserting markets…")
    inserted = insert_markets(all_markets, embs)
    print(f"[seed] upserted {inserted} markets.")




# ====== NEW: cycle bookkeeping ======

# helpers/supabase_db.py

def get_next_cycle_number() -> int:
    """
    Returns the next cycle_number by selecting the current max and adding 1.
    Uses a simple ordered select to avoid RPC.
    """
    sb = get_supabase()
    resp = (
        sb.table("cycle_logs")
        .select("cycle_number")
        .order("cycle_number", desc=True)
        .limit(1)
        .execute()
    )
    if resp.data and resp.data[0].get("cycle_number") is not None:
        return int(resp.data[0]["cycle_number"]) + 1
    return 1

def log_cycle_stats(stats: dict) -> None:
    sb = get_supabase()
    payload = {
        "cycle_number": stats.get("cycle_number"),
        "start_time": stats.get("start_time"),
        "end_time": stats.get("end_time"),
        "status": stats.get("status"),
        "tweets_fetched": stats.get("tweets_fetched", 0),
        "new_markets_fetched": stats.get("new_markets_fetched", 0),
        "correlations_found": stats.get("correlations_found", 0),
        "messages_sent": stats.get("messages_sent", 0),
        "notes": stats.get("notes"),
    }
    sb.table("cycle_logs").insert(payload).execute()

# ====== NEW: accounts to poll ======

def get_pollable_x_accounts() -> List[dict]:
    """
    Return active x_accounts that have at least one follower in user_x_follows.
    Fields: id (uuid), handle (text)
    """
    sb = get_supabase()
    # exists subquery via RPC-free approach: use two queries or a view; here do two-step
    accounts = (
        sb.table("x_accounts")
        .select("id, handle, is_active")
        .eq("is_active", True)
        .execute()
        .data
        or []
    )
    if not accounts:
        return []

    # Filter to those that have at least one follower
    ids = [a["id"] for a in accounts]
    # Fetch follows grouped
    follows = (
        sb.table("user_x_follows")
        .select("x_account_id")
        .in_("x_account_id", ids)
        .limit(1_000_000)
        .execute()
        .data
        or []
    )
    followed = set([f["x_account_id"] for f in follows])
    return [a for a in accounts if a["id"] in followed]

# helpers/supabase_db.py

def touch_accounts_last_checked(account_ids: List[str], ts_iso: str) -> None:
    """
    Update last_checked_at for existing x_accounts.
    Never inserts (avoids NULL handle issue).
    """
    if not account_ids:
        return
    sb = get_supabase()
    BATCH = 300
    for i in range(0, len(account_ids), BATCH):
        chunk = [aid for aid in account_ids[i:i+BATCH] if aid]  # guard empties
        if not chunk:
            continue
        # Update existing rows only
        sb.table("x_accounts") \
          .update({"last_checked_at": ts_iso}) \
          .in_("id", chunk) \
          .execute()

# ====== NEW: tweets upsert (batched) ======

# def insert_tweets_batched(normalized_tweets: List[dict]) -> int:
#     """
#     normalized_tweets fields required:
#       id, x_account_id, text, tweet_url, author_name, author_url, created_at_utc (ISO), embedding (optional)
#     """
#     if not normalized_tweets:
#         return 0

#     sb = get_supabase()
#     BATCH = 300
#     total = 0
#     for i in range(0, len(normalized_tweets), BATCH):
#         batch = normalized_tweets[i:i+BATCH]
#         # ensure embedding is list or None
#         for t in batch:
#             emb = t.get("embedding")
#             if hasattr(emb, "tolist"):
#                 t["embedding"] = emb.tolist()
#         sb.table("tweets").upsert(
#             batch,
#             on_conflict="id",
#             returning="minimal",
#         ).execute()
#         total += len(batch)
#     return total



# helpers/supabase_db.py


def insert_tweets_batched(normalized_tweets: list[dict]) -> int:
    """
    normalized_tweets required fields:
      id, x_account_id, text, tweet_url, author_name, author_url, created_at_utc (ISO)
    optional:
      embedding (list[float]) — if missing, we will generate
    """
    if not normalized_tweets:
        return 0

    # 1) Build a parallel list of texts to embed
    to_embed_idx = []
    to_embed_texts = []
    for idx, t in enumerate(normalized_tweets):
        emb = t.get("embedding")
        if emb is None or (hasattr(emb, "__len__") and len(emb) == 0):
            # sanitize text
            txt = (t.get("text") or "").replace("\n", " ").strip()
            if txt:
                to_embed_idx.append(idx)
                to_embed_texts.append(txt)

    # 2) Generate embeddings (batched) for those needing it
    if to_embed_texts:
        embs = generate_embeddings(to_embed_texts)  # returns list[np.ndarray]
        # 3) Patch them back on the rows
        for i, emb in zip(to_embed_idx, embs):
            normalized_tweets[i]["embedding"] = (
                emb.tolist() if hasattr(emb, "tolist") else emb
            )

    # 4) Upsert in chunks
    sb = get_supabase()
    BATCH = 300
    total = 0
    for i in range(0, len(normalized_tweets), BATCH):
        batch = normalized_tweets[i:i+BATCH]
        # Ensure JSON-serializable embedding
        for t in batch:
            emb = t.get("embedding")
            if hasattr(emb, "tolist"):
                t["embedding"] = emb.tolist()
        sb.table("tweets").upsert(
            batch,
            on_conflict="id",
            returning="minimal",
        ).execute()
        total += len(batch)
    return total

# helpers/supabase_db.py

# helpers/supabase_db.py

def normalize_handle(h: str) -> str:
    return (h or "").strip().lstrip("@").lower()

def ensure_x_accounts_for_handles(handles: list[str]) -> dict[str, str]:
    """
    Ensure each handle exists in x_accounts and return a mapping {handle_lower: id}.
    Skips empty/invalid handles to avoid NOT NULL violations.
    """
    sb = get_supabase()

    # sanitize & dedupe
    clean = []
    for h in handles or []:
        n = normalize_handle(h)
        if n:  # keep only non-empty after normalization
            clean.append(n)
    clean = sorted(set(clean))

    if not clean:
        return {}

    # 1) fetch existing
    existing_rows = (
        sb.table("x_accounts")
        .select("id, handle")
        .in_("handle", clean)
        .execute()
        .data or []
    )
    handle_to_id = {normalize_handle(r["handle"]): r["id"] for r in existing_rows if r.get("handle")}

    # 2) upsert missing (STRICTLY valid handles only)
    missing = [h for h in clean if h not in handle_to_id]
    if missing:
        payload = [{"handle": h, "is_active": True} for h in missing if h]  # guard!
        if payload:
            sb.table("x_accounts").upsert(
                payload,
                on_conflict="handle",
                returning="minimal",
            ).execute()

        # 3) reselect to get IDs (covers both existing + newly inserted)
        refreshed = (
            sb.table("x_accounts")
            .select("id, handle")
            .in_("handle", clean)
            .execute()
            .data or []
        )
        handle_to_id = {normalize_handle(r["handle"]): r["id"] for r in refreshed if r.get("handle")}

    return handle_to_id



def attach_x_account_ids_to_tweets(tweets: list[dict]) -> list[dict]:
    """
    For any tweet missing x_account_id, infer handle and attach the id.
    """
    # collect handles we can infer
    handles = []
    for t in tweets:
        if t.get("x_account_id"):
            continue
        h = t.get("_source_handle") or (t.get("author_url") or "").rsplit("/", 1)[-1]
        if h:
            handles.append(h)

    handle_to_id = ensure_x_accounts_for_handles(handles)
    # patch
    for t in tweets:
        if not t.get("x_account_id"):
            h = t.get("_source_handle") or (t.get("author_url") or "").rsplit("/", 1)[-1]
            hid = handle_to_id.get(normalize_handle(h))
            if hid:
                t["x_account_id"] = hid
    return tweets



def get_existing_market_ids(ids: list[str]) -> set[str]:
    if not ids:
        return set()
    sb = get_supabase()
    existing = (
        sb.table("markets")
        .select("id")
        .in_("id", ids)
        .execute()
        .data or []
    )
    return {row["id"] for row in existing if row.get("id")}



# helpers/supabase_db.py (append these)

def get_unprocessed_tweets_batch(limit: int = 40) -> list[dict]:
    sb = get_supabase()
    res = sb.table("tweets").select(
        "id,text,embedding,created_at_utc"
    ).eq("is_processed", False).order("created_at_utc", desc=False).limit(limit).execute()
    return res.data or []

def upsert_tweet_embeddings(id_to_emb: dict[str, list[float]]) -> None:
    if not id_to_emb:
        return
    sb = get_supabase()
    rows = [{"id": tid, "embedding": emb} for tid, emb in id_to_emb.items()]
    # upsert — only touching id + embedding
    sb.table("tweets").upsert(rows, on_conflict="id", returning="minimal").execute()

def rpc_match_markets(query_embedding: list[float], k: int = 50) -> list[dict]:
    sb = get_supabase()
    # res = sb.rpc("match_markets", {
    #     "query_embedding": query_embedding,
    #     "match_count": k
    # }).execute()
    res = sb.rpc("match_markets_v3", {
        "query_embedding": query_embedding,
        "match_count": k
    }).execute()
    return res.data or []

def upsert_correlations(rows: list[dict]) -> None:
    """
    rows require: tweet_id, market_id, relevance_score, relevance_reason,
    urgency_score, urgency_reason, engine_version, llm_model, tokens_input, tokens_output, latency_ms, metadata, llm_category_name
    """
    if not rows:
        return
    sb = get_supabase()
    sb.table("tweet_market_correlations").upsert(
        rows,
        on_conflict="tweet_id,market_id",
        returning="minimal",
    ).execute()

def mark_tweets_processed(ids: list[str]) -> None:
    if not ids:
        return
    sb = get_supabase()
    sb.table("tweets").update({"is_processed": True}).in_("id", ids).execute()