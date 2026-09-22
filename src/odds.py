"""
odds.py

Pulls current betting lines for upcoming UFC cards from bestfightodds.com
and upserts them into data/ufc_odds.csv, so the model's market blend (and
src.value) work on fights that haven't happened yet.

For each bout the line is the MEDIAN moneyline across the sportsbooks BFO
tracks — a poor man's consensus that's robust to one book's stale number.

BFO and ufcstats disagree on details (fighter spellings like "Zachary
Reese" vs "Zach Reese", event dates occasionally off by one day), so BFO
events are matched to the ufcstats schedule by date proximity and bouts by
normalized/fuzzy fighter names *within that one card* — and the rows are
written with the ufcstats names and event date, which is what everything
downstream (backtest joins, src.card lookups) keys on.

Usage:
    python3 -m src.odds             # refresh lines for all upcoming cards
    python3 -m src.odds --dry-run   # show what would be written

Typical flow before fight night:
    python3 -m src.odds && python3 -m src.card
"""
import csv
import difflib
import html as htmllib
import os
import re
import ssl
import statistics
import sys
import urllib.request
from datetime import datetime, timedelta

from src.scrape import _Session, parse_event_bouts, upcoming_events

BFO_URL = "https://www.bestfightodds.com/"
ODDS_COLUMNS = ["DATE", "RED", "BLUE", "RED_ODDS", "BLUE_ODDS",
                "RED_RANK", "BLUE_RANK", "EMPTY_ARENA"]


def fetch_bfo_page():
    """The BFO front page (all upcoming cards with odds are inlined on it)."""
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    req = urllib.request.Request(
        BFO_URL, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
    return urllib.request.urlopen(req, timeout=30, context=context).read().decode(
        "utf-8", errors="replace")


def _parse_bfo_date(text, today):
    """BFO shows "July 11th" with no year — pick the year that lands nearest."""
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text.strip())
    for year in (today.year, today.year + 1, today.year - 1):
        try:
            date = datetime.strptime(f"{cleaned} {year}", "%B %d %Y")
        except ValueError:
            return None
        if -14 <= (date - today).days <= 365:
            return date
    return None


def parse_bfo_events(page, today=None):
    """
    [{name, date, bouts: [(name_a, odds_a, name_b, odds_b), ...]}] for every
    event table on the BFO front page. Odds are the median across books;
    bouts missing a usable line on either side are dropped.
    """
    today = today or datetime.now()
    headers = [{"name": htmllib.unescape(m.group(1)).strip(),
                "date": _parse_bfo_date(m.group(2), today),
                "start": m.start()}
               for m in re.finditer(
                   r'<div class="table-header"><a href="/events/[^"]+"><h1>([^<]+)</h1></a>'
                   r'<span class="table-header-date">([^<]+)</span>', page)]

    events = []
    for i, header in enumerate(headers):
        end = headers[i + 1]["start"] if i + 1 < len(headers) else len(page)
        table_at = page.find('<table class="odds-table">', header["start"], end)
        if table_at < 0 or header["date"] is None:
            continue

        fighters = []  # (name, median_odds), in row order — consecutive rows pair up
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page[table_at:end], re.S):
            name_m = re.search(r'/fighters/[^"]+"><span class="t-b-fcc">([^<]+)</span>', row)
            if not name_m:
                continue  # prop bet / header row
            odds = [int(v) for v in re.findall(r'<span id="oID[^"]*">([+\-]\d+)</span>', row)]
            fighters.append((htmllib.unescape(name_m.group(1)).strip(),
                             int(statistics.median(odds)) if odds else None))

        bouts = [(a, ao, b, bo) for (a, ao), (b, bo) in zip(fighters[0::2], fighters[1::2])
                 if ao is not None and bo is not None]
        if bouts:
            events.append({"name": header["name"], "date": header["date"], "bouts": bouts})
    return events


def _norm(name):
    """Spelling-insensitive key: lowercase letters only ("Lone'er" == "Loneer")."""
    return re.sub(r"[^a-z]", "", name.lower())


def _match_name(target, candidates):
    """
    The candidate matching a BFO fighter name: exact on the normalized form,
    else the unique close fuzzy match ("Zachary Reese" -> "Zach Reese").
    """
    by_norm = {_norm(c): c for c in candidates}
    if _norm(target) in by_norm:
        return by_norm[_norm(target)]
    close = difflib.get_close_matches(_norm(target), list(by_norm), n=2, cutoff=0.75)
    return by_norm[close[0]] if len(close) == 1 else None


def match_lines(bfo_events, ufcstats_events, session, log=print):
    """
    Joins BFO odds onto the ufcstats schedule. Returns CSV-ready rows keyed
    by ufcstats names/dates: one dict per bout that matched on both sides.
    """
    rows = []
    for event in ufcstats_events:
        try:
            event_date = datetime.strptime(event["date"], "%B %d, %Y")
        except ValueError:
            continue
        nearby = [b for b in bfo_events
                  if abs((b["date"] - event_date).days) <= 1 and b["name"].startswith("UFC")]
        if not nearby:
            continue

        card = parse_event_bouts(session.get(event["url"]))
        matched = 0
        for bfo in nearby:
            for name_a, odds_a, name_b, odds_b in bfo["bouts"]:
                for bout in card:
                    if len(bout["fighters"]) != 2:
                        continue
                    ma = _match_name(name_a, bout["fighters"])
                    mb = _match_name(name_b, bout["fighters"])
                    if ma and mb and ma != mb:
                        rows.append({"DATE": event_date.strftime("%Y-%m-%d"),
                                     "RED": ma, "BLUE": mb,
                                     "RED_ODDS": float(odds_a), "BLUE_ODDS": float(odds_b),
                                     "RED_RANK": "", "BLUE_RANK": "", "EMPTY_ARENA": ""})
                        matched += 1
                        break
        log(f"  {event['name']} ({event['date']}): lines for {matched}/{len(card)} bouts")
    return rows


def upsert_odds(data_dir, rows):
    """
    Merges rows into ufc_odds.csv: replaces the line for a (date, pairing)
    already present (lines move), prepends anything new. Returns (updated,
    added) counts.
    """
    path = f"{data_dir}/ufc_odds.csv"
    with open(path, newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))

    def key(row):
        return (row["DATE"], frozenset([row["RED"].strip().lower(), row["BLUE"].strip().lower()]))

    incoming = {key(r): r for r in rows}
    updated = 0
    for i, row in enumerate(existing):
        new = incoming.pop(key(row), None)
        if new is not None:
            existing[i] = new
            updated += 1

    merged = list(incoming.values()) + existing  # CSV is newest-first
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ODDS_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
    return updated, len(incoming)


def refresh_odds(data_dir, dry_run=False, log=print):
    session = _Session()
    log("fetching schedule from ufcstats and lines from bestfightodds...")
    bfo = parse_bfo_events(fetch_bfo_page())
    schedule = upcoming_events(session)
    rows = match_lines(bfo, schedule, session, log=log)

    if dry_run:
        for r in rows:
            log(f"  {r['DATE']}  {r['RED']} {r['RED_ODDS']:+.0f}  vs  {r['BLUE']} {r['BLUE_ODDS']:+.0f}")
        log(f"dry run — would upsert {len(rows)} lines")
        return len(rows)

    updated, added = upsert_odds(data_dir, rows)
    log(f"odds updated: {updated} lines refreshed, {added} added")
    return len(rows)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    refresh_odds(data_dir, dry_run="--dry-run" in sys.argv)
