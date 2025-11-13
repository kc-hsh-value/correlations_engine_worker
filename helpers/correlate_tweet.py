# file: scripts/correlate_tweet.py
from __future__ import annotations

import argparse
import json

from semantic_correlation import correlate_tweet_id

def main():
    ap = argparse.ArgumentParser(description="Correlate a tweet to markets and emit JSON.")
    ap.add_argument("tweet_id", help="tweet id in your 'tweets' table")
    ap.add_argument("--top-n", type=int, default=50, help="top-K semantic candidates (default 50)")
    ap.add_argument("--relevance-threshold", type=float, default=0.6, help="min relevance to keep (default 0.6)")
    ap.add_argument("--pretty", action="store_true", help="pretty print JSON")
    args = ap.parse_args()

    result = correlate_tweet_id(
        tweet_id=args.tweet_id,
        top_n=args.top_n,
        relevance_threshold=args.relevance_threshold,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()