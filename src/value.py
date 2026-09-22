"""
value.py — compare the model's prediction against a posted betting line.

Usage:
    python3 -m src.value "fighter a" "fighter b" ODDS_A ODDS_B

    ODDS_A / ODDS_B are American odds for each fighter, e.g.:
    python3 -m src.value "max holloway" "dustin poirier" -150 +130

Reports each side's market-implied probability (devigged), the model's
probability, the disagreement (edge), and the expected value of a $1 bet.

HONEST CAVEAT, from backtesting this exact model against ~1,000 historical
closing lines: the market is efficient. Raw model/market disagreements LOST
money at every threshold; only very large edges (EV > 0.15) broke even, and
that result is within noise. Treat the output as analysis of where the model
and market disagree — not as betting advice.
"""
import sys

from src.backtest import _american_odds_to_prob, build_current_states
from src.model import load_model, predict_proba

EDGE_NOTEWORTHY = 0.15  # backtested minimum EV where bets stopped losing money


def _payout(odds):
    """Profit on a winning $1 bet at American odds."""
    return odds / 100 if odds > 0 else 100 / -odds


def analyze(name_a, name_b, odds_a, odds_b, data_dir="data"):
    model = load_model()
    if model is None:
        raise SystemExit("No trained model found — run: python3 -m src.model")

    name_a, name_b = name_a.strip().lower(), name_b.strip().lower()
    states, physical, pool_means = build_current_states(data_dir)
    for name in (name_a, name_b):
        if name not in physical:
            raise SystemExit(f"fighter not found: '{name}'")

    # Devig the posted line into fair market probabilities.
    p_raw_a, p_raw_b = _american_odds_to_prob(odds_a), _american_odds_to_prob(odds_b)
    market_a = p_raw_a / (p_raw_a + p_raw_b)
    market_b = 1 - market_a

    # The market line is blended with the stats-only model's probability
    # (see model.fit_blend); any remaining disagreement is driven by the
    # fight stats.
    model_a = predict_proba(model, states, physical, pool_means, name_a, name_b,
                            odds_prob_diff=market_a - market_b)
    model_b = 1 - model_a

    print(f"{'':22s}{name_a.title():>18s}{name_b.title():>18s}")
    print(f"{'posted odds':22s}{odds_a:>+18.0f}{odds_b:>+18.0f}")
    print(f"{'market probability':22s}{market_a:>17.1%}{market_b:>17.1%}")
    print(f"{'model probability':22s}{model_a:>17.1%}{model_b:>17.1%}")
    print(f"{'edge (model-market)':22s}{model_a - market_a:>+17.1%}{model_b - market_b:>+17.1%}")

    ev_a = model_a * _payout(odds_a) - (1 - model_a)
    ev_b = model_b * _payout(odds_b) - (1 - model_b)
    print(f"{'EV per $1 bet':22s}{ev_a:>+17.2f}{ev_b:>+17.2f}")
    print()

    best_side, best_ev = (name_a, ev_a) if ev_a >= ev_b else (name_b, ev_b)
    if best_ev > EDGE_NOTEWORTHY:
        print(f"NOTEWORTHY DISAGREEMENT: model sees value on {best_side.title()} "
              f"(EV {best_ev:+.2f}/$1). Backtest note: even this bucket only "
              f"broke even historically — treat as analysis, not advice.")
    elif best_ev > 0:
        print(f"Mild disagreement on {best_side.title()} (EV {best_ev:+.2f}/$1) — "
              f"below the {EDGE_NOTEWORTHY:.2f} threshold that backtesting says is noise. Pass.")
    else:
        print("Model agrees with the market: no positive-EV side.")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    analyze(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
