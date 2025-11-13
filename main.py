# main.py
from __future__ import annotations
import asyncio
from datetime import datetime, timezone, timedelta
import discord
import socket

from helpers.supabase_db import (
    prune_expired_markets,
    get_markets,
    insert_markets,
    get_next_cycle_number,
    log_cycle_stats,
    insert_tweets_batched,
    touch_accounts_last_checked,
    attach_x_account_ids_to_tweets,
    get_existing_market_ids
)
from helpers.embeddings import generate_embeddings
# If your file is helpers/x_fetch.py use this import:
from helpers.x import fetch_recent_tweets_for_accounts, normalize_tweets
from helpers.correlation_engine import run_correlation_engine_parallel
# If your file is helpers/x.py keep: from helpers.x import ...

# ============ config ============
CHECK_INTERVAL_SECONDS = 900   # 15 minutes
TWEET_LOOKBACK_MIN = 45        # lookback window; can set to 30 to match interval

async def network_health_check() -> bool:
    try:
        await asyncio.get_event_loop().getaddrinfo('google.com', 80)
        return True
    except (socket.gaierror, OSError):
        return False

async def alpha_cycle_loop(bot_client: discord.Client | None = None):
    print("--- Starting Tweet-Market Correlation Agent (Supabase) ---")
    print(f"Polling every {CHECK_INTERVAL_SECONDS} seconds…")

    while True:
        if not await network_health_check():
            print("!!! Network health check failed. Sleeping 10min.")
            await asyncio.sleep(600)
            continue

        cycle = {
            "cycle_number": get_next_cycle_number(),
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "status": "PENDING",
            "tweets_fetched": 0,
            "new_markets_fetched": 0,
            "correlations_found": 0,
            "messages_sent": 0,
            "notes": None,
        }

        try:
            print("\n" + "="*60)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- Cycle #{cycle['cycle_number']} ---")

            # 1) Maintain market cache (prune)
            print("[1/3] Pruning expired markets…")
            # pruned = prune_expired_markets()
            # if pruned:
            #     print(f"  - pruned {pruned}")
            try:
                pruned = prune_expired_markets()
                if pruned:
                    print(f"  - pruned {pruned}")
            except Exception as e:
                print(f"  - prune skipped due to error: {e}")

            # 2) Fetch new markets since last interval, embed, upsert
            print("[2/3] Fetch new markets since last interval…")
            since_time = datetime.now(timezone.utc) - timedelta(seconds=CHECK_INTERVAL_SECONDS)
            since_str = since_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            new_markets = get_markets(start_date_min=since_str)
            cycle["new_markets_fetched"] = len(new_markets) if new_markets else 0

            if new_markets:
                ids = [m.get("id") for m in new_markets if m.get("id")]
                existing_ids = get_existing_market_ids(ids)
                fresh = [m for m in new_markets if m.get("id") not in existing_ids]

                print(f"  - fetched {len(new_markets)} markets, {len(fresh)} are new (not in DB)")

                if fresh:
                    texts = [f"Question: {m.get('question','')}\nDescription: {m.get('description','')}" for m in fresh]
                    embs = generate_embeddings(texts)
                    inserted = insert_markets(fresh, embs)
                    print(f"  - inserted {inserted} new markets")
                else:
                    print("  - nothing new to insert (skipping embed/upsert)")

            # 3) Fetch tweets for pollable accounts (last N minutes), normalize, store
            print("[3/3] Fetch recent tweets…")
            raw_tweets = fetch_recent_tweets_for_accounts(TWEET_LOOKBACK_MIN)
            cycle["tweets_fetched"] = len(raw_tweets) if raw_tweets else 0

            if raw_tweets:
                normalized = normalize_tweets(raw_tweets)
                if normalized:
                    # attach x_account_id (creates accounts as needed), then filter to safe rows
                    normalized = attach_x_account_ids_to_tweets(normalized)
                    ready = [t for t in normalized if t.get("x_account_id")]
                    skipped = len(normalized) - len(ready)
                    if skipped:
                        print(f"[tweets] skipping {skipped} tweets with unresolved handle/x_account_id")

                    if ready:
                        inserted_tweets = insert_tweets_batched(ready)
                        print(f"  - upserted {inserted_tweets} tweets")

                        # update last_checked only for known IDs (update-only helper)
                        acc_ids = list({t["x_account_id"] for t in ready if t.get("x_account_id")})
                        if acc_ids:
                            touch_accounts_last_checked(acc_ids, datetime.now(timezone.utc).isoformat())

            print("\n[4/4] Correlating…")
            # run_correlation_engine()
            run_correlation_engine_parallel(stream_ndjson=True)  # prints NDJSON lines as tweets finish
            cycle["status"] = "SUCCESS"
            print("--- Cycle finished (ingestion only). ---")

        except Exception as e:
            print(f"!! CRITICAL ERROR in cycle #{cycle['cycle_number']}: {e}")
            cycle["status"] = "FAILED"
            cycle["notes"] = str(e)

        finally:
            cycle["end_time"] = datetime.now(timezone.utc).isoformat()
            log_cycle_stats(cycle)
            print(f"--- Cycle #{cycle['cycle_number']} status: {cycle['status']} ---")
            print(f"Sleeping {CHECK_INTERVAL_SECONDS} seconds…")
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)