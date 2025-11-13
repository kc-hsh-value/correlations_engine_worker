# helpers/x_fetch.py
from __future__ import annotations
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List

from helpers.supabase_db import get_pollable_x_accounts

API_KEY = os.getenv("X_API_KEY")  # twitterapi.io key
SEARCH_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

# Hardcoded fallback (used only if no pollable accounts in DB)
FALLBACK_ACCOUNTS = [
    "unusual_whales", "WatcherGuru", "spectatorindex", "Polymarket", "whalewatchpoly",
    "rawsalerts", "DeItaone", "PolymarketIntel", "Megatron_ron", "GlobeEyeNews",
    "Currentreport1", "DropSiteNews", "disclosetv", "zoomerfied", "AFpost"
]

def _fmt(dt: datetime) -> str:
    # twitterapi.io expects YYYY-MM-DD_HH:MM:SS_UTC
    return dt.strftime("%Y-%m-%d_%H:%M:%S_UTC")

def fetch_recent_tweets_for_accounts(minutes_ago: int) -> List[Dict]:
    """
    Fetch recent tweets from twitterapi.io for all pollable accounts from Supabase.
    Falls back to a hardcoded list if no accounts have followers yet.
    Returns raw tweet dicts from the API, with two extras:
      - _source_handle
      - _x_account_id (if known)
    """
    if not API_KEY:
        print("[x_fetch] X_API_KEY missing; skipping tweet fetch.")
        return []

    # Pull accounts that have at least one follower (our poll set)
    pollable = get_pollable_x_accounts()  # [{id, handle}, ...]
    if not pollable:
        # fallback to hardcoded list for bootstrap
        pollable = [{"id": None, "handle": h, "is_active": True} for h in FALLBACK_ACCOUNTS]
        print(f"[x_fetch] no pollable accounts found; using fallback list of {len(pollable)} handles.")

    until_time = datetime.now(timezone.utc)
    since_time = until_time - timedelta(minutes=minutes_ago)
    since_str = _fmt(since_time)
    until_str = _fmt(until_time)

    headers = {"X-API-Key": API_KEY}
    all_tweets: List[Dict] = []

    print(f"[x_fetch] fetching since {since_str} until {until_str} for {len(pollable)} accounts…")

    for acc in pollable:
        handle = acc["handle"]
        query = f"from:{handle} since:{since_str} until:{until_str} include:nativeretweets"
        params = {"query": query, "queryType": "Latest"}

        next_cursor = None
        while True:
            if next_cursor:
                params["cursor"] = next_cursor
            try:
                resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                tweets_page = data.get("tweets", [])
                if tweets_page:
                    for t in tweets_page:
                        t["_source_handle"] = handle
                        t["_x_account_id"] = acc.get("id")  # may be None for fallback
                    all_tweets.extend(tweets_page)

                if data.get("has_next_page") and data.get("next_cursor"):
                    next_cursor = data.get("next_cursor")
                else:
                    break
            except requests.exceptions.RequestException as e:
                print(f"[x_fetch] error fetching @{handle}: {e}")
                break

            time.sleep(0.2)  # be polite

    print(f"[x_fetch] total raw tweets: {len(all_tweets)}")
    return all_tweets


from datetime import datetime, timezone
from typing import List, Dict, Optional

def _parse_tweet_created_at(v: Optional[str]) -> str:
    """
    Accepts: "Wed Sep 03 13:15:01 +0000 2025" (Twitter), ISO strings, or None.
    Returns ISO 8601 UTC string.
    """
    if not v:
        return datetime.now(timezone.utc).isoformat()
    v = v.strip()
    # Twitter format: "Wed Sep 03 13:15:01 +0000 2025"
    for fmt in ('%a %b %d %H:%M:%S %z %Y', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f%z'):
        try:
            dt = datetime.strptime(v, fmt)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    # last resort: try fromisoformat after normalizing Z
    try:
        v2 = v.replace('Z', '+00:00')
        dt = datetime.fromisoformat(v2)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()

def normalize_tweets(raw_tweets: List[Dict]) -> List[Dict]:
    """
    Output fields expected by your DB layer:
      id, x_account_id, text, tweet_url, author_name, author_url,
      author_handle, author_avatar_url, published_at, created_at_utc, is_processed, full_data_json
    """
    normalized: List[Dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for t in raw_tweets:
        try:
            tid = str(t.get("id") or t.get("tweet_id"))
            if not tid:
                continue

            # URLs
            url = (
                t.get("url")
                or t.get("tweet_url")
                or t.get("twitterUrl")  # present in your sample
                or f"https://twitter.com/i/web/status/{tid}"
            )

            # Text
            txt = t.get("text") or ""

            # Author block (twitterapi.io uses `author`; sometimes `user`)
            author = t.get("author") or t.get("user") or {}
            handle = (
                author.get("userName") or author.get("screen_name")
                or author.get("username") or t.get("_source_handle")
            )
            name = author.get("name") or t.get("author_name") or handle or "Unknown"
            avatar = (
                author.get("profilePicture")
                or author.get("profile_image_url_https")
                or None
            )
            author_url = (
                author.get("url")
                or author.get("twitterUrl")
                or (f"https://twitter.com/{handle}" if handle else None)
            )

            # Published time from API
            published_src = (
                t.get("createdAt") or  # camelCase (your sample)
                t.get("created_at") or
                t.get("created_at_utc")
            )
            published_at = _parse_tweet_created_at(published_src)

            normalized.append({
                "id": tid,
                "x_account_id": t.get("_x_account_id"),  # may be None for fallback
                "text": txt,
                "tweet_url": url,
                "author_name": name,
                # "author_handle": handle,
                # "author_avatar_url": avatar,
                "author_url": author_url,
                "published_at": published_at,    # <-- NEW: true tweet publish time
                "created_at_utc": now_iso,       # <-- ingestion time (keep for auditing)
                "is_processed": False,
                "full_data_json": t,
            })
        except Exception as e:
            print(f"[x_fetch] skip malformed tweet: {e}")
            continue

    return normalized

