"""
ledger.py

A running, gradeable record of the model's real picks — the honest
scoreboard the backtest can't be. Every `python3 -m src.card` run logs its
picks here (upserted, so re-running before fight night just refreshes them);
after results land via `python3 -m src.scrape`, grading locks each pick
against what actually happened.

    python3 -m src.ledger           # grade pending picks + show the record

Over time this builds a live out-of-sample track record: pick accuracy,
probability quality (log-loss), how the market did on the same fights, and
the return of a hypothetical flat $1 bet on every pick at the logged odds.

The ledger lives in data/predictions.csv. A pick is "pending" until its
WINNER column is filled by grading; graded rows are never overwritten.
"""
import csv
import math
import os
import sys
from datetime import datetime

LEDGER_COLUMNS = [
    "LOGGED",                     # when the pick was made (YYYY-MM-DD HH:MM)
    "EVENT", "DATE",              # event name and date (YYYY-MM-DD)
    "FIGHTER_A", "FIGHTER_B",
    "P_A",                        # final P(A wins) used for the pick (blended if line)
    "MODEL_P_A", "MARKET_P_A",    # stats-only and devigged-market P(A); market may be blank
    "ODDS_A", "ODDS_B",           # American odds at log time (blank without a line)
    "PICK", "PICK_PROB",          # the pick and its probability
    "PICK_METHOD", "METHOD_PROB", # likely method, conditional on the pick winning
    # -- filled by grading --
    "WINNER",                     # actual winner name, or "draw"/"nc"
    "METHOD",                     # actual method string from the results CSV
    "CORRECT",                    # 1/0 (blank for draw/nc)
]


def _ledger_path(data_dir):
    return os.path.join(data_dir, "predictions.csv")


def _read(data_dir):
    path = _ledger_path(data_dir)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(data_dir, rows):
    with open(_ledger_path(data_dir), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _key(row):
    return (row["DATE"], frozenset([row["FIGHTER_A"].strip().lower(),
                                    row["FIGHTER_B"].strip().lower()]))


def log_picks(data_dir, picks):
    """
    Upserts pick rows (dicts with the pre-grading LEDGER_COLUMNS) into the
    ledger. A pending pick for the same bout is replaced by the fresh one;
    graded rows are left alone. Returns (updated, added).
    """
    rows = _read(data_dir)
    incoming = {}
    for p in picks:
        row = {c: "" for c in LEDGER_COLUMNS}
        row.update(p)
        incoming[_key(row)] = row

    updated = 0
    for i, row in enumerate(rows):
        new = incoming.pop(_key(row), None)
        if new is not None and not row["WINNER"]:
            rows[i] = new
            updated += 1

    rows.extend(incoming.values())
    _write(data_dir, rows)
    return updated, len(incoming)


def _load_results(data_dir):
    """{(date_str, {name, name}): (winner_or_draw/nc, method)} from the results CSV."""
    from src.backtest import _load_event_dates

    event_dates = _load_event_dates(data_dir)
    results = {}
    with open(f"{data_dir}/ufc_fight_results.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bout, outcome = row["BOUT"], row["OUTCOME"].strip()
            if " vs. " not in bout or "/" not in outcome:
                continue
            a, b = (n.strip() for n in bout.split(" vs. "))
            date = event_dates.get(row["EVENT"].strip())
            if date is None:
                continue
            a_result = outcome.split("/")[0]
            winner = a if a_result == "W" else b if a_result == "L" else \
                "draw" if a_result == "D" else "nc"
            results[(date.strftime("%Y-%m-%d"), frozenset([a.lower(), b.lower()]))] = \
                (winner, row["METHOD"].strip())
    return results


def grade(data_dir, log=print):
    """Fills WINNER/METHOD/CORRECT on pending picks that now have a result."""
    rows = _read(data_dir)
    results = _load_results(data_dir)
    graded = 0
    for row in rows:
        if row["WINNER"]:
            continue
        hit = results.get(_key(row))
        if hit is None:
            continue
        winner, method = hit
        row["WINNER"] = winner
        row["METHOD"] = method
        if winner not in ("draw", "nc"):
            row["CORRECT"] = "1" if winner.lower() == row["PICK"].strip().lower() else "0"
        graded += 1
    if graded:
        _write(data_dir, rows)
    log(f"graded {graded} new pick(s)")
    return graded


def _flat_bet_return(row):
    """Profit of $1 on the pick at the logged odds; None without a line."""
    odds = row["ODDS_A"] if row["PICK"] == row["FIGHTER_A"] else row["ODDS_B"]
    try:
        odds = float(odds)
    except ValueError:
        return None
    if row["CORRECT"] == "0":
        return -1.0
    return odds / 100 if odds > 0 else 100 / -odds


def report(data_dir, log=print):
    """Prints the running record: accuracy, probability quality, market, P/L."""
    rows = _read(data_dir)
    if not rows:
        log("ledger is empty — run `python3 -m src.card` to log picks")
        return

    done = [r for r in rows if r["CORRECT"] in ("0", "1")]
    pending = [r for r in rows if not r["WINNER"]]
    other = len(rows) - len(done) - len(pending)  # draws / no-contests

    log(f"\nledger: {len(rows)} picks — {len(done)} graded, {len(pending)} pending"
        + (f", {other} draw/NC" if other else ""))

    if pending:
        stale = [r for r in pending
                 if (datetime.now() - datetime.strptime(r["DATE"], "%Y-%m-%d")).days > 14]
        upcoming = sorted({(r["DATE"], r["EVENT"]) for r in pending if r not in stale})
        for date, event in upcoming:
            n = sum(1 for r in pending if r["EVENT"] == event)
            log(f"  pending: {event} ({date}) — {n} picks")
        if stale:
            log(f"  {len(stale)} pick(s) >2 weeks old with no result — scratched fight?")

    if not done:
        return

    correct = [r for r in done if r["CORRECT"] == "1"]
    probs = [(float(r["PICK_PROB"]), int(r["CORRECT"])) for r in done]
    ll = -sum(c * math.log(max(p, 1e-12)) + (1 - c) * math.log(max(1 - p, 1e-12))
              for p, c in probs) / len(probs)
    log(f"\n  picks:   {len(correct)}/{len(done)} correct ({len(correct)/len(done):.1%})   "
        f"log-loss {ll:.4f}")

    lined = [r for r in done if r["MARKET_P_A"]]
    if lined:
        market_correct = sum(
            (float(r["MARKET_P_A"]) >= 0.5) == (r["WINNER"].lower() == r["FIGHTER_A"].lower())
            for r in lined)
        log(f"  market:  {market_correct}/{len(lined)} correct "
            f"({market_correct/len(lined):.1%}) on the same lined fights")
        returns = [v for v in (_flat_bet_return(r) for r in lined) if v is not None]
        if returns:
            log(f"  flat $1 on every lined pick: "
                f"{sum(returns):+.2f} over {len(returns)} bets "
                f"({sum(returns)/len(returns):+.1%} ROI)")

    method_rows = [r for r in done if r["CORRECT"] == "1" and r["PICK_METHOD"] and r["METHOD"]]
    if method_rows:
        from src.model import method_class, METHOD_NAMES
        hits = sum(METHOD_NAMES[method_class(r["METHOD"])] == r["PICK_METHOD"]
                   for r in method_rows if method_class(r["METHOD"]) is not None)
        log(f"  method:  {hits}/{len(method_rows)} right about HOW, when the pick won")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    grade(data_dir)
    report(data_dir)
