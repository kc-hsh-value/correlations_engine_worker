#!/usr/bin/env python3
"""
Polymarket price history checker

Examples
--------
# Last month, hourly fidelity (recommended):
python price_chart.py \
  --yes 104173557214744537570424345347209544585775842950109756851652855913015295701992 \
  --no  44528029102356085806317866371026691780796471200782980570839327755136990994869 \
  --interval 1m --fidelity 60

# Custom window (UTC epoch seconds):
python price_chart.py --yes <YES_TOKEN> --no <NO_TOKEN> --start 1735689600 --end 1738281600 --fidelity 60

# Dry-run chart with synthetic data (no network):
python price_chart.py --mock
"""

import argparse
import csv
from datetime import datetime, timezone
import sys
from typing import List, Dict, Optional

import requests
import matplotlib.pyplot as plt


BASE_URL = "https://clob.polymarket.com/prices-history"


def fetch_history(
    token_id: str,
    *,
    interval: Optional[str] = None,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    fidelity: Optional[int] = None,
    _retry_no_fidelity: bool = True,
) -> List[Dict]:
    """
    Fetch {t,p} points for a token. Use either `interval` OR (`start_ts` & `end_ts`).
    Retries once without fidelity on HTTP 400 (API quirk).
    """
    if not token_id:
        return []

    params = {"market": token_id}
    if interval:
        params["interval"] = interval
    else:
        if start_ts is not None:
            params["startTs"] = str(start_ts)
        if end_ts is not None:
            params["endTs"] = str(end_ts)

    if fidelity is not None:
        params["fidelity"] = str(fidelity)

    r = requests.get(BASE_URL, params=params, timeout=30)
    if r.status_code == 400 and fidelity is not None and _retry_no_fidelity:
        # retry once without fidelity
        return fetch_history(
            token_id,
            interval=interval,
            start_ts=start_ts,
            end_ts=end_ts,
            fidelity=None,
            _retry_no_fidelity=False,
        )
    r.raise_for_status()

    js = r.json() or {}
    hist = js.get("history", [])
    out = []
    for row in hist:
        t = row.get("t")
        p = row.get("p")
        if t is None or p is None:
            continue
        out.append({"t": int(t), "p": float(p)})
    return out


def dt_from_epoch(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def plot_series(yes_hist: List[Dict], no_hist: List[Dict], title: str, out_jpg: str):
    x_yes = [dt_from_epoch(pt["t"]) for pt in yes_hist]
    y_yes = [pt["p"] for pt in yes_hist]
    x_no  = [dt_from_epoch(pt["t"]) for pt in no_hist]
    y_no  = [pt["p"] for pt in no_hist]

    plt.figure(figsize=(12, 6), dpi=140)
    plt.plot(x_yes, y_yes, label="YES")
    plt.plot(x_no,  y_no,  label="NO")
    plt.title(title)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Midpoint price ($)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_jpg, format="jpg")
    plt.close()


def write_csv(path: str, rows: List[Dict], header=("t","p")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get("t"), r.get("p")])


def mock_series(n=300, start_ts: Optional[int] = None, step_sec=3600):
    """
    Simple synthetic curves (sine-ish wiggle) to test plotting without network.
    """
    import math, random
    if start_ts is None:
        start_ts = int(datetime.now(timezone.utc).timestamp()) - n*step_sec
    yes = []
    no  = []
    b = random.random() * 0.2
    for i in range(n):
        t = start_ts + i*step_sec
        y = 0.5 + 0.15*math.sin(i/12.0) + b
        y = max(0.01, min(0.99, y))
        nprice = 1.0 - y + (random.random()-0.5)*0.01
        nprice = max(0.01, min(0.99, nprice))
        yes.append({"t": t, "p": y})
        no.append({"t": t, "p": nprice})
    return yes, no


def main():
    ap = argparse.ArgumentParser(description="Polymarket price history chart")
    ap.add_argument("--yes", help="YES tokenId (CLOB)", default="")
    ap.add_argument("--no",  help="NO  tokenId (CLOB)", default="")
    ap.add_argument("--interval", choices=["1m","1w","1d","6h","1h","max"], help="duration window ending now")
    ap.add_argument("--start", type=int, help="startTs (epoch seconds UTC)")
    ap.add_argument("--end",   type=int, help="endTs   (epoch seconds UTC)")
    ap.add_argument("--fidelity", type=int, help="resolution in minutes", default=60)
    ap.add_argument("--out", default="polymarket_price_check.jpg", help="output JPG path")
    ap.add_argument("--csv-prefix", default="polymarket_prices", help="prefix for CSV dumps")
    ap.add_argument("--mock", action="store_true", help="generate synthetic data (no network)")
    args = ap.parse_args()

    if args.mock:
        yes_hist, no_hist = mock_series()
        title = "Mock price history (synthetic)"
    else:
        if not (args.interval or (args.start is not None and args.end is not None)):
            print("ERROR: provide either --interval OR both --start/--end", file=sys.stderr)
            sys.exit(1)
        if not args.yes and not args.no:
            print("ERROR: provide at least one token (--yes or --no)", file=sys.stderr)
            sys.exit(1)

        yes_hist = fetch_history(
            args.yes,
            interval=args.interval,
            start_ts=args.start,
            end_ts=args.end,
            fidelity=args.fidelity,
        ) if args.yes else []

        no_hist = fetch_history(
            args.no,
            interval=args.interval,
            start_ts=args.start,
            end_ts=args.end,
            fidelity=args.fidelity,
        ) if args.no else []

        win_desc = f"interval={args.interval}" if args.interval else f"{args.start}→{args.end}"
        title = f"Polymarket price history ({win_desc}, fidelity={args.fidelity})"

    # Write CSVs for debugging/verification
    write_csv(f"{args.csv_prefix}_yes.csv", yes_hist)
    write_csv(f"{args.csv_prefix}_no.csv",  no_hist)

    # Plot & save
    plot_series(yes_hist, no_hist, title, args.out)

    # Console summary
    def first_last(arr):
        return (arr[0] if arr else None, arr[-1] if arr else None)

    print("YES points:", len(yes_hist))
    print("NO  points:", len(no_hist))
    f, l = first_last(yes_hist)
    if f and l:
        print("YES range:", dt_from_epoch(f["t"]), "→", dt_from_epoch(l["t"]))
    f, l = first_last(no_hist)
    if f and l:
        print("NO  range:", dt_from_epoch(f["t"]), "→", dt_from_epoch(l["t"]))
    print("Saved chart:", args.out)
    print("CSV dumps  :", f"{args.csv_prefix}_yes.csv", f"{args.csv_prefix}_no.csv")


if __name__ == "__main__":
    main()