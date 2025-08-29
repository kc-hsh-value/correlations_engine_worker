# helpers/supabase_db.py
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from helpers.supabase_client import get_supabase

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

def prune_expired_markets(now_utc: Optional[datetime] = None) -> int:
    """
    Marks markets as inactive if end_date_utc < now().
    Returns number of rows updated.
    """
    sb = get_supabase()
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    # PostgREST expects ISO8601
    now_iso = now_utc.isoformat()

    resp = (
        sb.table("markets")
        .update({"is_active": False})
        .lt("end_date_utc", now_iso)
        .eq("is_active", True)
        .execute()
    )
    # supabase-py returns affected rows in data length (no count). Best effort:
    return len(resp.data or [])

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