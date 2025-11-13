# file: scripts/correlate_tweets_parallel.py
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from typing import Any, Dict, List

# Import your existing function (adjust path if needed)
from semantic_correlation import correlate_tweet_id


def _p95(xs: List[int] | List[float] | None):
    if not xs:
        return None
    xs = sorted(xs)
    i = int(math.ceil(0.95 * len(xs))) - 1
    return xs[i]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_one(tweet_id: str, top_n: int, relevance_threshold: float) -> Dict[str, Any]:
    """
    Run correlation for a single tweet, measure wall time, and wrap result.
    Never raises; errors are captured into the returned dict.
    """
    started = time.perf_counter()
    started_at = _utc_now_iso()

    try:
        result = correlate_tweet_id(
            tweet_id=tweet_id,
            top_n=top_n,
            relevance_threshold=relevance_threshold,
        )
        # Drop candidates to avoid console spam
        if isinstance(result, dict) and "candidates" in result:
            result = {k: v for k, v in result.items() if k != "candidates"}
        ok = True
        err = None
    except Exception as e:
        result = None
        ok = False
        err = f"{type(e).__name__}: {e}"

    wall_ms = int((time.perf_counter() - started) * 1000)
    finished_at = _utc_now_iso()

    out: Dict[str, Any] = {
        "tweet_id": tweet_id,
        "ok": ok,
        "wall_ms": wall_ms,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if ok:
        out["result"] = result
    else:
        out["error"] = err
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Correlate multiple tweets to markets in parallel and emit JSON."
    )
    ap.add_argument(
        "tweet_ids",
        nargs="+",
        help="one or more tweet IDs from your 'tweets' table",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="top-K semantic candidates per tweet (default 50)",
    )
    ap.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.6,
        help="min relevance to keep (default 0.6)",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="number of worker threads (default 2)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="batch wall-clock timeout in seconds (optional)",
    )
    ap.add_argument(
        "--pretty",
        action="store_true",
        help="pretty-print JSON",
    )
    args = ap.parse_args()

    overall_started = time.perf_counter()
    started_at = _utc_now_iso()

    results_by_id: Dict[str, Dict[str, Any]] = {}

    # Launch all tasks
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures_map = {
            ex.submit(_run_one, tid, args.top_n, args.relevance_threshold): tid
            for tid in args.tweet_ids
        }

        # Wait for all or timeout; stop on first exception if you prefer
        done, not_done = wait(
            futures_map.keys(),
            timeout=args.timeout,
            return_when=FIRST_EXCEPTION if args.timeout is None else None,
        )

        # Collect finished
        for fut in done:
            tid = futures_map[fut]
            try:
                results_by_id[tid] = fut.result()
            except Exception as e:
                results_by_id[tid] = {
                    "tweet_id": tid,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "started_at": started_at,
                    "finished_at": _utc_now_iso(),
                    "wall_ms": None,
                }

        # Mark unfinished as timeouts
        for fut in not_done:
            tid = futures_map[fut]
            fut.cancel()
            results_by_id[tid] = {
                "tweet_id": tid,
                "ok": False,
                "error": "TimeoutError: batch timeout",
                "started_at": started_at,
                "finished_at": _utc_now_iso(),
                "wall_ms": None,
            }

    # Preserve input order
    results: List[Dict[str, Any]] = [results_by_id[tid] for tid in args.tweet_ids]

    # Aggregate latency rollups
    wall_ms_vals: List[int] = [r["wall_ms"] for r in results if r.get("wall_ms") is not None]
    vector_ms: List[int] = []
    llm_ms: List[int] = []
    for r in results:
        t = (r.get("result") or {}).get("timing_ms") or {}
        if "vector_rpc" in t:
            vector_ms.append(t["vector_rpc"])
        if "llm" in t:
            llm_ms.append(t["llm"])

    meta_rollups = {
        "wall_ms": {
            "avg": int(sum(wall_ms_vals) / len(wall_ms_vals)) if wall_ms_vals else None,
            "p95": _p95(wall_ms_vals),
            "max": max(wall_ms_vals) if wall_ms_vals else None,
        },
        "vector_rpc_ms": {
            "avg": int(sum(vector_ms) / len(vector_ms)) if vector_ms else None,
            "p95": _p95(vector_ms),
        },
        "llm_ms": {
            "avg": int(sum(llm_ms) / len(llm_ms)) if llm_ms else None,
            "p95": _p95(llm_ms),
        },
    }

    overall_wall_ms = int((time.perf_counter() - overall_started) * 1000)
    finished_at = _utc_now_iso()

    payload = {
        "meta": {
            "started_at": started_at,
            "finished_at": finished_at,
            "total_wall_ms": overall_wall_ms,
            "num_tweets": len(args.tweet_ids),
            "max_workers": args.max_workers,
            "top_n": args.top_n,
            "relevance_threshold": args.relevance_threshold,
            "latency": meta_rollups,
            "timeout_sec": args.timeout,
        },
        "results": results,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()