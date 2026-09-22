"""
scrape.py

Refreshes the CSVs in data/ with everything that happened on ufcstats.com
since the last run: new events, their fight results, round-by-round fight
stats, and any fighters we haven't seen before (details, physical attributes,
and career record).

The site fronts each new visitor with a small JavaScript proof-of-work
challenge (hash a nonce until it meets a difficulty target, POST the answer,
receive a cookie). _Session solves it the same way a browser would and then
reuses the cookie. Requests are rate-limited to ~1/second to be polite.

Usage:
    python3 -m src.scrape             # fetch new events and update data/
    python3 -m src.scrape --dry-run   # show what would be added, write nothing

After a refresh, retrain so the model sees the new fights:
    python3 -m src.model

Only ufcstats.com data is refreshed. data/ufc_odds.csv comes from a different
source and is not touched — fights without a line degrade gracefully (the
model simply skips the market blend for them).
"""
import csv
import hashlib
import html as htmllib
import http.cookiejar
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

BASE = "http://ufcstats.com"
COMPLETED_URL = f"{BASE}/statistics/events/completed?page=all"
UPCOMING_URL = f"{BASE}/statistics/events/upcoming"

RESULT_COLUMNS = ["EVENT", "BOUT", "OUTCOME", "WEIGHTCLASS", "METHOD", "ROUND",
                  "TIME", "TIME FORMAT", "REFEREE", "DETAILS", "URL"]
STATS_COLUMNS = ["EVENT", "BOUT", "ROUND", "FIGHTER", "KD", "SIG.STR.", "SIG.STR. %",
                 "TOTAL STR.", "TD", "TD %", "SUB.ATT", "REV.", "CTRL",
                 "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]


class _Session:
    """Cookie-carrying fetcher that answers the site's proof-of-work check."""

    def __init__(self, delay=1.0):
        self.delay = delay
        jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self.opener.addheaders = [("User-Agent", "ufc-predictor data refresh (personal project)")]

    def _read(self, url):
        return self.opener.open(url, timeout=30).read().decode("utf-8", errors="replace")

    def _solve_challenge(self, page):
        """Answers the interstitial: find n with sha256(nonce:n) under target."""
        m = re.search(r'nonce="([0-9a-f]+)".*?Array\((\d+)\+1\)', page, re.S)
        if not m:
            return False
        nonce, zeros = m.group(1), int(m.group(2))
        n = 0
        while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith("0" * zeros):
            n += 1
        req = urllib.request.Request(
            f"{BASE}/__c", data=f"nonce={nonce}&n={n}".encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.opener.open(req, timeout=30)
        return True

    def get(self, url, tries=3):
        last_err = None
        for _ in range(tries):
            try:
                page = self._read(url)
                if "Checking your browser" in page and self._solve_challenge(page):
                    page = self._read(url)
                time.sleep(self.delay)
                return page
            except OSError as e:
                last_err = e
                time.sleep(self.delay * 3)
        raise RuntimeError(f"failed to fetch {url}: {last_err}")


def _text(fragment):
    """Tag-stripped, entity-unescaped, whitespace-collapsed text of an HTML fragment."""
    return htmllib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()


def _labelled_value(flat_page, label):
    """
    The text after an <i class=...__label>Label:</i> marker on a fight page.
    Values sit either directly after the label or inside one wrapper tag
    (<span> for Referee, <i style=...> for Method). Expects flattened HTML.
    """
    m = re.search(re.escape(label) + r":\s*</i>\s*(?:<[a-z]+[^>]*>\s*)?([^<]*)", flat_page)
    return htmllib.unescape(m.group(1)).strip() if m else ""


def parse_events_list(page):
    """[{url, name, date, location}] from an events listing, newest first."""
    events = []
    pattern = (r'<a href="(http://ufcstats\.com/event-details/[0-9a-f]+)"[^>]*>\s*([^<]+?)\s*</a>'
               r'.*?<span class="b-statistics__date">\s*([^<]+?)\s*</span>'
               r'\s*</i>\s*</td>\s*<td[^>]*>\s*([^<]+?)\s*</td>')
    for m in re.finditer(pattern, page, re.S):
        events.append({"url": m.group(1), "name": htmllib.unescape(m.group(2)).strip(),
                       "date": m.group(3).strip(), "location": htmllib.unescape(m.group(4)).strip()})
    return events


def parse_event_bouts(page):
    """
    The fights listed on an event page, top (main event) first:
    [{url, fighters: [name, name], weightclass, completed}].
    Upcoming events use the same table but without result flags.
    """
    bouts = []
    for row in page.split('<tr class="b-fight-details__table-row')[1:]:
        m = re.search(r'fight-details/([0-9a-f]+)', row)
        if not m:
            continue
        names = [htmllib.unescape(n).strip() for n in
                 re.findall(r'fighter-details/[0-9a-f]+[\'"]?>\s*([^<]+?)\s*</a>', row)]
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        weightclass = _text(cells[6]) if len(cells) > 6 else ""
        bouts.append({
            "url": f"{BASE}/fight-details/{m.group(1)}",
            "fighters": names[:2],
            "weightclass": weightclass,
            "completed": "b-flag__text" in row,
        })
    return bouts


def _round_rows(table_html):
    """
    Per-round stat cells from a fight page's per-round table:
    {round_label: [cells_fighter_1, cells_fighter_2]} where each cells list
    holds the column texts for that fighter ("15 of 25", "60%", "0:36", ...).
    """
    rounds = {}
    segments = re.split(r"<thead[^>]*>\s*<th[^>]*>\s*(Round \d+)\s*</th>\s*</thead>", table_html)
    for i in range(1, len(segments), 2):
        label, body = segments[i], segments[i + 1]
        tr = re.search(r"<tr[^>]*>(.*?)</tr>", body, re.S)
        if not tr:
            continue
        per_fighter = [[], []]
        for td in re.findall(r"<td[^>]*>(.*?)</td>", tr.group(1), re.S):
            texts = [_text(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", td, re.S)]
            if len(texts) >= 2:
                per_fighter[0].append(texts[0])
                per_fighter[1].append(texts[1])
        rounds[label] = per_fighter
    return rounds


def parse_fight(page, event_name, fight_url):
    """
    One completed fight page -> (results_row, stats_rows, fighter_urls).
    results_row/stats_rows are dicts keyed like the CSV columns.
    Returns (None, [], []) if the fight has no result yet.
    """
    statuses = re.findall(r"b-fight-details__person-status[^>]*>\s*([A-Z]{1,2})\s*<", page)
    names = [htmllib.unescape(n).strip() for n in
             re.findall(r"b-fight-details__person-link[^>]*>\s*([^<]+?)\s*</a>", page)]
    if len(statuses) != 2 or len(names) != 2 or not all(s in ("W", "L", "D", "NC") for s in statuses):
        return None, [], []

    bout = f"{names[0]} vs. {names[1]}"
    title_m = re.search(r"b-fight-details__fight-title[^>]*>(.*?)</i>", page, re.S)

    flat = re.sub(r"\s+", " ", page)
    details_m = re.search(r"Details:\s*</i>(.*?)</p>", flat)
    results_row = {
        "EVENT": event_name,
        "BOUT": bout,
        "OUTCOME": "/".join(statuses),
        "WEIGHTCLASS": _text(title_m.group(1)) if title_m else "",
        "METHOD": _labelled_value(flat, "Method"),
        "ROUND": _labelled_value(flat, "Round"),
        "TIME": _labelled_value(flat, "Time"),
        "TIME FORMAT": _labelled_value(flat, "Time format"),
        "REFEREE": _labelled_value(flat, "Referee"),
        "DETAILS": _text(details_m.group(1)) if details_m else "",
        "URL": fight_url,
    }

    # Fight pages carry four tables: totals, per-round totals, significant
    # strikes, per-round significant strikes. The CSV wants the per-round
    # ones, joined into one row per fighter per round.
    stats_rows = []
    tables = re.findall(r"<table[^>]*>(.*?)</table>", page, re.S)
    if len(tables) == 4:
        totals, sig = _round_rows(tables[1]), _round_rows(tables[3])
        for round_label, tot in totals.items():
            sig_round = sig.get(round_label)
            for side in (0, 1):
                # totals: Fighter, KD, Sig.str, Sig%, Total, Td, Td%, Sub, Rev, Ctrl
                # sig:    Fighter, Sig.str, Sig%, Head, Body, Leg, Distance, Clinch, Ground
                t = tot[side]
                s = sig_round[side] if sig_round else [""] * 9
                if len(t) < 10 or len(s) < 9:
                    continue
                stats_rows.append({
                    "EVENT": event_name, "BOUT": bout, "ROUND": round_label,
                    "FIGHTER": t[0], "KD": t[1], "SIG.STR.": t[2], "SIG.STR. %": t[3],
                    "TOTAL STR.": t[4], "TD": t[5], "TD %": t[6], "SUB.ATT": t[7],
                    "REV.": t[8], "CTRL": t[9],
                    "HEAD": s[3], "BODY": s[4], "LEG": s[5],
                    "DISTANCE": s[6], "CLINCH": s[7], "GROUND": s[8],
                })

    fighter_urls = [f"{BASE}/fighter-details/{fid}" for fid in
                    dict.fromkeys(re.findall(r"fighter-details/([0-9a-f]+)", page))]
    return results_row, stats_rows, fighter_urls


def parse_fighter(page, url):
    """One fighter page -> rows for the details, tott, and records CSVs."""
    flat = re.sub(r"\s+", " ", page)
    name_m = re.search(r"b-content__title-highlight\">\s*([^<]+?)\s*</span>", flat)
    full_name = htmllib.unescape(name_m.group(1)).strip() if name_m else ""
    first, _, last = full_name.partition(" ")

    nick_m = re.search(r"b-content__Nickname\">\s*([^<]*?)\s*</p>", flat)
    nickname = htmllib.unescape(nick_m.group(1)).strip() if nick_m else ""

    def attr(label):
        m = re.search(re.escape(label) + r":\s*</i>\s*([^<]*)", flat)
        return htmllib.unescape(m.group(1)).strip() if m else "--"

    record_m = re.search(r"Record:\s*(\d+)-(\d+)-(\d+)", flat)
    wins, losses, draws = record_m.groups() if record_m else ("0", "0", "0")

    return {
        "details": {"FIRST": first, "LAST": last, "NICKNAME": nickname, "URL": url},
        "tott": {"FIGHTER": full_name, "HEIGHT": attr("Height"), "WEIGHT": attr("Weight"),
                 "REACH": attr("Reach"), "STANCE": attr("STANCE"), "DOB": attr("DOB"),
                 "URL": url},
        "record": {"NAME": full_name, "WINS": wins, "LOSSES": losses, "DRAWS": draws},
    }


def _csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _prepend_csv(path, columns, new_rows):
    """Inserts new_rows after the header — these CSVs are newest-first."""
    _write_csv(path, columns, new_rows + _csv_rows(path))


def _parse_event_date(text):
    try:
        return datetime.strptime(text.strip(), "%B %d, %Y")
    except ValueError:
        return None


def refresh(data_dir, dry_run=False, log=print):
    """
    Fetches everything new from ufcstats.com and updates the CSVs in data_dir.
    Returns a summary dict with counts of what was added.
    """
    session = _Session()

    known_events = {r["URL"] for r in _csv_rows(f"{data_dir}/ufc_event_details.csv")}
    known_fighters = {r["URL"] for r in _csv_rows(f"{data_dir}/ufc_fighter_details.csv")}

    listing = parse_events_list(session.get(COMPLETED_URL))
    new_events = [e for e in listing if e["url"] not in known_events]
    log(f"events on ufcstats not yet in data/: {len(new_events)}")

    event_rows, result_rows, stats_rows = [], [], []
    new_fighter_urls = []
    for event in new_events:  # newest first, matching the CSVs' order
        bouts = parse_event_bouts(session.get(event["url"]))
        if not any(b["completed"] for b in bouts):
            log(f"  skipping (not fought yet): {event['name']} — {event['date']}")
            continue
        log(f"  scraping: {event['name']} — {event['date']} ({len(bouts)} bouts)")
        event_rows.append({"EVENT": event["name"], "URL": event["url"],
                           "DATE": event["date"], "LOCATION": event["location"]})
        for bout in bouts:
            if not bout["completed"]:
                continue
            result, stats, fighter_urls = parse_fight(session.get(bout["url"]), event["name"], bout["url"])
            if result is None:
                continue
            result_rows.append(result)
            stats_rows.extend(stats)
            new_fighter_urls.extend(u for u in fighter_urls if u not in known_fighters)

    details_rows, tott_rows, record_rows = [], [], []
    for url in dict.fromkeys(new_fighter_urls):
        parsed = parse_fighter(session.get(url), url)
        log(f"  new fighter: {parsed['tott']['FIGHTER']}")
        details_rows.append(parsed["details"])
        tott_rows.append(parsed["tott"])
        record_rows.append(parsed["record"])

    summary = {"events": len(event_rows), "fights": len(result_rows),
               "stat_rows": len(stats_rows), "fighters": len(details_rows)}
    if dry_run:
        log(f"dry run — would add: {summary}")
        return summary

    if event_rows:
        _prepend_csv(f"{data_dir}/ufc_event_details.csv",
                     ["EVENT", "URL", "DATE", "LOCATION"], event_rows)
        _prepend_csv(f"{data_dir}/ufc_fight_results.csv", RESULT_COLUMNS, result_rows)
        _prepend_csv(f"{data_dir}/ufc_fight_stats.csv", STATS_COLUMNS, stats_rows)
    if details_rows:
        with open(f"{data_dir}/ufc_fighter_details.csv", "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["FIRST", "LAST", "NICKNAME", "URL"]).writerows(details_rows)
        with open(f"{data_dir}/ufc_fighter_tott.csv", "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["FIGHTER", "HEIGHT", "WEIGHT", "REACH",
                                          "STANCE", "DOB", "URL"]).writerows(tott_rows)
        with open(f"{data_dir}/ufc_fighter_records.csv", "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["NAME", "WINS", "LOSSES", "DRAWS"]).writerows(record_rows)

    log(f"added: {summary}")
    if summary["fights"]:
        log("retrain to pick up the new fights:  python3 -m src.model")
    return summary


def upcoming_events(session=None):
    """[{url, name, date, location}] for scheduled events, soonest first."""
    return parse_events_list((session or _Session()).get(UPCOMING_URL))


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    refresh(data_dir, dry_run="--dry-run" in sys.argv)
