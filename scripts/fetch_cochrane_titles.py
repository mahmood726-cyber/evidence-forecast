"""Batch-fetch Cochrane review titles from CrossRef for all DOIs in
cache/review_dois.txt. CrossRef allows polite concurrency; default 8
parallel workers with 0.2s pacing stays well under 50 req/sec guidance.

Output: cache/cochrane_titles.csv with columns doi, title, container,
publisher (for provenance).
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOIS_IN = ROOT / "cache" / "review_dois.txt"
TITLES_OUT = ROOT / "cache" / "cochrane_titles.csv"

UA = "EvidenceForecast/0.1 (research tool; mailto:mahmood.ahmad2@nhs.net)"


def fetch_one(doi: str) -> dict:
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data.get("message", {})
        title_list = msg.get("title", [])
        title = title_list[0] if title_list else ""
        container_list = msg.get("container-title", [])
        container = container_list[0] if container_list else ""
        publisher = msg.get("publisher", "")
        return {"doi": doi, "title": title, "container": container,
                "publisher": publisher, "error": ""}
    except urllib.error.HTTPError as e:
        return {"doi": doi, "title": "", "container": "", "publisher": "",
                "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"doi": doi, "title": "", "container": "", "publisher": "",
                "error": str(e)[:120]}


def main() -> int:
    if not DOIS_IN.exists():
        print(f"missing {DOIS_IN}; run extract_dois.R first")
        return 1
    dois = [line.strip() for line in DOIS_IN.read_text().splitlines() if line.strip()]
    print(f"fetching {len(dois)} Cochrane review titles from CrossRef...")

    # Resume: skip DOIs already cached
    cached = {}
    if TITLES_OUT.exists():
        with TITLES_OUT.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cached[row["doi"]] = row
        print(f"  resume: {len(cached)} already cached")

    results = list(cached.values())
    todo = [d for d in dois if d not in cached]
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, d): d for d in todo}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)} fetched")
            time.sleep(0.05)  # extra politeness

    TITLES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TITLES_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doi", "title", "container", "publisher", "error"])
        w.writeheader()
        for r in sorted(results, key=lambda r: r["doi"]):
            w.writerow(r)

    n_ok = sum(1 for r in results if r["title"])
    n_err = sum(1 for r in results if r["error"])
    print(f"done: {n_ok} titles, {n_err} errors -> {TITLES_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
