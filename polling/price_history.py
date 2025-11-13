# polling/price_history.py
import os
import time
import json
import math
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from supabase import create_client, Client  # pip install supabase

load_dotenv()

CLOB_URL = "https://clob.polymarket.com/prices-history"

# ---- Tunables ---------------------------------------------------------------
MAX_WINDOW_DAYS = 9            # keep < 10 days for CLOB API limits
SLEEP_BETWEEN_CALLS = 0.15     # polite backoff between calls
REQUEST_INTERVAL = "1h"        # align to hourly buckets
BUCKET_SEC = 3600              # 1 hour buckets (chart/DB expect this)
FORWARD_FILL = True            # fill missing YES/NO with last seen price in-bucket
PRINT_DEBUG = True
# -----------------------------------------------------------------------------


def sb() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def to_iso(sec: int) -> str:
    return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()


def epoch_from_iso(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def floor_bucket(ts_s: int, bucket_s: int = BUCKET_SEC) -> int:
    return (ts_s // bucket_s) * bucket_s


def split_windows(start_ts: int, end_ts: int, max_days: int = MAX_WINDOW_DAYS) -> List[Tuple[int, int]]:
    out = []
    cur = start_ts
    step = max_days * 24 * 3600
    while cur < end_ts:
        nxt = min(cur + step, end_ts)
        out.append((cur, nxt))
        cur = nxt
    return out


# ---- market discovery / metadata -------------------------------------------

def get_correlated_market_ids(client: Client) -> List[str]:
    """All unique market IDs that have at least one tweet correlation (paginated)."""
    ids: set[str] = set()
    from_idx, page = 0, 5000
    while True:
        res = client.table("tweet_market_correlations").select("market_id").range(from_idx, from_idx + page - 1).execute()
        rows = res.data or []
        for r in rows:
            mid = r.get("market_id")
            if mid:
                ids.add(str(mid))
        if not rows or len(rows) < page:
            break
        from_idx += page
    return sorted(ids)


def fetch_market_meta(client: Client, market_id: str) -> Optional[Dict]:
    res = client.table("markets").select("id, question, full_data_json").eq("id", market_id).single().execute()
    return res.data if res.data else None


def parse_tokens_from_full_data(fdj: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract clobTokenIds for YES/NO in outcomes order."""
    if not fdj:
        return (None, None)

    def as_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return []
        return []

    outcomes = as_list(fdj.get("outcomes") or fdj.get("market", {}).get("outcomes"))
    clobs = as_list(fdj.get("clobTokenIds") or fdj.get("market", {}).get("clobTokenIds"))

    yi = next((i for i, o in enumerate(outcomes) if str(o).strip().lower() == "yes"), None)
    ni = next((i for i, o in enumerate(outcomes) if str(o).strip().lower() == "no"), None)

    yes = clobs[yi] if yi is not None and yi < len(clobs) else None
    no  = clobs[ni] if ni is not None and ni < len(clobs) else None
    return (str(yes) if yes else None, str(no) if no else None)


def parse_market_times(fdj: Dict) -> Tuple[Optional[int], Optional[int]]:
    """Find plausible created/closed timestamps."""
    def to_epoch(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            try:
                return epoch_from_iso(v)
            except Exception:
                try:
                    return int(v)
                except Exception:
                    return None
        return None

    created = fdj.get("createdAt") or fdj.get("market", {}).get("createdAt") or fdj.get("openTime") or fdj.get("startTime")
    closed  = fdj.get("closedAt")  or fdj.get("market", {}).get("closedAt")  or fdj.get("closeTime") or fdj.get("endTime")
    return to_epoch(created), to_epoch(closed)


# ---- CLOB fetch (windowed, retry) ------------------------------------------

def prices_history_chunk(token_id: str, start_ts: int, end_ts: int, interval: str = REQUEST_INTERVAL, include_fidelity: bool = False) -> List[Dict]:
    """
    Fetch a window of ticks. We keep it simple for hourly: no fidelity by default.
    If API returns 400, retry once without fidelity (if we had included).
    """
    if not token_id:
        return []

    params = {
        "market": token_id,
        "startTs": str(start_ts),
        "endTs": str(end_ts),
        "interval": interval,
    }
    if include_fidelity:
        params["fidelity"] = "10"

    resp = requests.get(CLOB_URL, params=params, timeout=30)
    if not resp.ok and include_fidelity and resp.status_code == 400:
        # fallback try once without fidelity
        resp = requests.get(CLOB_URL, params={**params, **{"interval": interval}}, timeout=30)

    resp.raise_for_status()
    data = resp.json() or {}
    hist = data.get("history", [])
    out = []
    for row in hist:
        t = row.get("t")
        p = row.get("p")
        if t is None or p is None:
            continue
        out.append({"t": int(t), "p": float(p)})
    return out


def prices_history_windowed(token_id: str, start_ts: int, end_ts: int) -> List[Dict]:
    if not token_id or not (start_ts < end_ts):
        return []

    windows = split_windows(start_ts, end_ts, MAX_WINDOW_DAYS)
    all_rows: List[Dict] = []
    for (ws, we) in windows:
        try:
            chunk = prices_history_chunk(token_id, ws, we, REQUEST_INTERVAL, include_fidelity=False)
            all_rows.extend(chunk)
        except requests.HTTPError as e:
            if PRINT_DEBUG:
                print(f"    [warn] chunk {ws}->{we} {e}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    # dedupe by timestamp (keep last)
    seen: Dict[int, Dict] = {}
    for r in all_rows:
        seen[r["t"]] = r
    return [seen[t] for t in sorted(seen.keys())]


# ---- bucketing / merging ----------------------------------------------------

def bucket_series(rows: List[Dict], bucket_s: int = BUCKET_SEC) -> Dict[int, float]:
    """
    Keep the **latest** tick per bucket. Return {bucket_ts: price}
    """
    buckets: Dict[int, Dict[str, int | float]] = {}
    for r in rows:
        t = int(r["t"])
        p = float(r["p"])
        b = floor_bucket(t, bucket_s)
        cur = buckets.get(b)
        if not cur or t > cur["t"]:  # keep latest in bucket
            buckets[b] = {"t": t, "p": p}
    return {b: float(v["p"]) for b, v in buckets.items()}


def merge_buckets(yes_b: Dict[int, float], no_b: Dict[int, float]) -> Dict[int, Dict[str, Optional[float]]]:
    all_buckets = sorted(set(yes_b.keys()) | set(no_b.keys()))
    out: Dict[int, Dict[str, Optional[float]]] = {}
    for b in all_buckets:
        out[b] = {"yes": yes_b.get(b), "no": no_b.get(b)}
    return out


def forward_fill_points(points: Dict[int, Dict[str, Optional[float]]]) -> Dict[int, Dict[str, Optional[float]]]:
    last_yes: Optional[float] = None
    last_no: Optional[float] = None
    for t in sorted(points.keys()):
        row = points[t]
        if row.get("yes") is not None:
            last_yes = row["yes"]
        else:
            row["yes"] = last_yes
        if row.get("no") is not None:
            last_no = row["no"]
        else:
            row["no"] = last_no
    return points


# ---- DB helpers -------------------------------------------------------------

def get_last_stored_bucket(client: Client, market_id: str) -> Optional[int]:
    """
    Get the last stored ts (as epoch seconds) and floor to bucket.
    """
    res = client.table("market_price_history") \
        .select("ts") \
        .eq("market_id", market_id) \
        .order("ts", desc=True) \
        .limit(1) \
        .execute()
    rows = res.data or []
    if not rows:
        return None
    last_iso = rows[0]["ts"]
    last_s = epoch_from_iso(last_iso)
    return floor_bucket(last_s, BUCKET_SEC)


def upsert_price_rows(client: Client, market_id: str, points: Dict[int, Dict[str, Optional[float]]]) -> int:
    """
    Upsert bucketed rows. 'ts' must be the **bucket start** (hour-aligned).
    Requires a UNIQUE constraint on (market_id, ts).
    """
    rows = []
    for t, d in points.items():
        rows.append({
            "market_id": market_id,
            "ts": to_iso(t),
            "yes_price": d.get("yes"),
            "no_price": d.get("no"),
            "source": "polymarket_clob",
        })
    if not rows:
        return 0
    client.table("market_price_history").upsert(rows, on_conflict="market_id,ts").execute()
    return len(rows)


# ---- ingest one market ------------------------------------------------------

def ingest_market_history(client: Client, market_id: str) -> None:
    meta = fetch_market_meta(client, market_id)
    if not meta:
        if PRINT_DEBUG:
            print(f"[skip] market {market_id} missing meta")
        return

    fdj = meta.get("full_data_json") or {}
    yes_token, no_token = parse_tokens_from_full_data(fdj)
    if not yes_token and not no_token:
        if PRINT_DEBUG:
            print(f"[skip] {market_id}: no clobTokenIds in full_data_json")
        return

    created, closed = parse_market_times(fdj)
    now_s = int(datetime.now(timezone.utc).timestamp())
    start_s = created or now_s - 30 * 24 * 3600   # if unknown, backfill ~30d
    end_s   = min(closed or now_s, now_s)

    # resume from last stored bucket + 1 hour
    last_bucket = get_last_stored_bucket(client, market_id)
    if last_bucket is not None and last_bucket >= start_s:
        start_s = min(max(last_bucket + BUCKET_SEC, start_s), end_s)

    if start_s >= end_s:
        if PRINT_DEBUG:
            print(f"[ok] {market_id}: up to date")
        return

    if PRINT_DEBUG:
        print(f"[ingest] {market_id} {to_iso(start_s)} → {to_iso(end_s)}")

    total = 0
    for (w_start, w_end) in split_windows(start_s, end_s, MAX_WINDOW_DAYS):
        # Fetch raw series
        yes_raw = prices_history_windowed(yes_token, w_start, w_end) if yes_token else []
        no_raw  = prices_history_windowed(no_token,  w_start, w_end) if no_token  else []

        if PRINT_DEBUG:
            print(f"  window {to_iso(w_start)} → {to_iso(w_end)}  raw yes:{len(yes_raw)} no:{len(no_raw)}")

        # Bucket to 1h (keep latest tick per bucket)
        yes_b = bucket_series(yes_raw, BUCKET_SEC)
        no_b  = bucket_series(no_raw,  BUCKET_SEC)

        if PRINT_DEBUG:
            print(f"          bucketed yes:{len(yes_b)} no:{len(no_b)} (bucket={BUCKET_SEC}s)")

        merged = merge_buckets(yes_b, no_b)
        if FORWARD_FILL:
            merged = forward_fill_points(merged)

        inserted = upsert_price_rows(client, market_id, merged)
        total += inserted

        if PRINT_DEBUG:
            print(f"          upserted rows: {inserted}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    if PRINT_DEBUG:
        print(f"[done] {market_id}: inserted/updated {total} rows")


# ---- main -------------------------------------------------------------------

def main():
    client = sb()

    # Strongly recommended once (outside script):
    # ALTER TABLE public.market_price_history ADD CONSTRAINT mph_unique UNIQUE (market_id, ts);

    market_ids = get_correlated_market_ids(client)
    print(f"[discover] {len(market_ids)} markets with ≥1 tweet correlation")

    for mid in market_ids:
        try:
            ingest_market_history(client, mid)
        except Exception as e:
            print(f"[err] {mid}: {e}")


if __name__ == "__main__":
    main()