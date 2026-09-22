"""
card.py

Predicts every fight on an upcoming UFC card, straight from ufcstats.com's
schedule — the natural "product" of the model: run it before fight night and
get a pick, win probability, and likely method for each bout.

Usage:
    python3 -m src.card             # the next scheduled event
    python3 -m src.card --list      # show all scheduled events
    python3 -m src.card 3           # the 3rd-next event
    python3 -m src.card mcgregor    # first event whose name matches

Fights involving someone with no UFC history and no known pre-UFC record
(usually a debutant the dataset hasn't seen) are listed without a prediction.
"""
import os
import sys
from datetime import datetime

from src.backtest import _find_odds, _load_odds, build_current_states
from src.ledger import log_picks
from src.model import load_model, predict_proba, predict_method_proba
from src.scrape import _Session, parse_event_bouts, upcoming_events

METHOD_LABELS = {"ko": "KO/TKO", "sub": "submission", "dec": "decision"}


def _pick_event(events, arg):
    if arg is None:
        return events[0]
    if arg.isdigit():
        index = int(arg) - 1
        if 0 <= index < len(events):
            return events[index]
        sys.exit(f"only {len(events)} events are scheduled")
    for event in events:
        if arg.lower() in event["name"].lower():
            return event
    sys.exit(f"no upcoming event matches {arg!r}")


def _predictable(name, states, physical):
    """Mirrors predictor._predict_ml's rule for who the model can rate."""
    if name not in physical:
        return False
    state = states[name]
    if state["wins"] + state["losses"] + state["draws"] > 0:
        return True
    phys = physical[name]
    return phys.get("pre_ufc_wins", 0) + phys.get("pre_ufc_losses", 0) > 0


def predict_card(event, session, data_dir, log=print):
    """Prints a prediction for every bout on an event page."""
    bouts = parse_event_bouts(session.get(event["url"]))
    if not bouts:
        log("no bouts announced yet for this event")
        return

    model = load_model()
    if model is None:
        sys.exit("no trained model found — run: python3 -m src.model")
    states, physical, pool_means = build_current_states(data_dir)

    # Betting lines, if src.odds has been run for this card: enables the
    # market blend and shows the market's own probability for comparison.
    from src.backtest import _parse_date
    odds = _load_odds(data_dir)
    event_date = _parse_date(event["date"])

    width = 74
    log("=" * width)
    log(f"{event['name']:^{width}}")
    log(f"{event['date'] + ' — ' + event['location']:^{width}}")
    log("=" * width)

    picks = []  # ledger rows for this card
    for bout in bouts:
        if len(bout["fighters"]) != 2:
            continue
        name_a, name_b = bout["fighters"]
        key_a, key_b = name_a.strip().lower(), name_b.strip().lower()

        log(f"\n  {bout['weightclass']}")
        log(f"  {name_a}  vs.  {name_b}")

        if not (_predictable(key_a, states, physical) and _predictable(key_b, states, physical)):
            log("    no prediction — not enough history for at least one fighter")
            continue

        line_info = _find_odds(odds, event_date, key_a, key_b)
        p_model = predict_proba(model, states, physical, pool_means, key_a, key_b)
        if line_info is not None:
            odds_prob_diff = line_info["prob"][key_a] - line_info["prob"][key_b]
            p_a = predict_proba(model, states, physical, pool_means, key_a, key_b,
                                odds_prob_diff=odds_prob_diff)
        else:
            p_a = p_model
        fav, p_fav, fav_key = (name_a, p_a, "a") if p_a >= 0.5 else (name_b, 1 - p_a, "b")

        line = f"    pick: {fav}  ({p_fav:.0%})"
        if line_info is not None:
            fav_lower = fav.strip().lower()
            line += f"  [market: {line_info['prob'][fav_lower]:.0%}]"
        best = best_p = None
        methods = predict_method_proba(model, states, physical, pool_means, key_a, key_b)
        if methods:
            fav_methods = methods[fav_key]
            total = sum(fav_methods.values())
            if total:
                best = max(fav_methods, key=fav_methods.get)
                best_p = fav_methods[best] / total
                line += f", most likely by {METHOD_LABELS[best]} ({best_p:.0%} of their wins)"
        log(line)

        picks.append({
            "LOGGED": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "EVENT": event["name"], "DATE": event_date.strftime("%Y-%m-%d"),
            "FIGHTER_A": name_a, "FIGHTER_B": name_b,
            "P_A": f"{p_a:.4f}", "MODEL_P_A": f"{p_model:.4f}",
            "MARKET_P_A": f"{line_info['prob'][key_a]:.4f}" if line_info else "",
            "ODDS_A": line_info["american"][key_a] if line_info else "",
            "ODDS_B": line_info["american"][key_b] if line_info else "",
            "PICK": fav, "PICK_PROB": f"{p_fav:.4f}",
            "PICK_METHOD": best or "", "METHOD_PROB": f"{best_p:.4f}" if best_p else "",
        })

    if picks:
        updated, added = log_picks(data_dir, picks)
        log(f"\nledger: {added} pick(s) logged, {updated} refreshed "
            f"(grade after the event with: python3 -m src.ledger)")

    log("\n" + "=" * width)
    log("picks with [market: ...] blend the model with the betting line (run")
    log("python3 -m src.odds to refresh lines); picks without one are stats-only.")
    log("method split is conditional on that fighter winning.")


def main(argv):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    session = _Session()
    events = upcoming_events(session)
    if not events:
        sys.exit("no upcoming events found")

    if "--list" in argv:
        for i, event in enumerate(events, 1):
            print(f"{i:2d}. {event['date']:<20} {event['name']}  ({event['location']})")
        return

    event = _pick_event(events, argv[0] if argv else None)
    predict_card(event, session, data_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
