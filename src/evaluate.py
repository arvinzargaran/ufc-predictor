"""
evaluate.py

Chronological holdout evaluation of the trained ensemble: train on the past,
test on the most recent fights, and report probability quality — not just
accuracy. At this model's accuracy plateau (~68%), real improvements show up
as better-calibrated probabilities (lower log-loss / Brier), and as closing
the gap to the betting market on specific segments.

Usage:
    python3 -m src.evaluate [test_size]     # default: last 1300 fights
"""
import math
import os
import sys

from src.backtest import FEATURE_NAMES, generate_dataset


def _col(name):
    return FEATURE_NAMES.index(name)


def _metrics(probs, actuals):
    """(accuracy, log_loss, brier) for P(A wins) vs 1/0 outcomes."""
    n = len(probs)
    if n == 0:
        return None
    eps = 1e-12
    acc = sum((p >= 0.5) == (a == 1) for p, a in zip(probs, actuals)) / n
    ll = -sum(a * math.log(max(p, eps)) + (1 - a) * math.log(max(1 - p, eps))
              for p, a in zip(probs, actuals)) / n
    brier = sum((p - a) ** 2 for p, a in zip(probs, actuals)) / n
    return acc, ll, brier


def _fmt(label, m, n):
    if m is None:
        return f"  {label:<28s} (no fights)"
    acc, ll, brier = m
    return f"  {label:<28s} acc {acc:6.1%}   log-loss {ll:.4f}   brier {brier:.4f}   (n={n})"


def _market_prob(row_meta):
    """Devigged P(A wins) from the American odds in a meta row, or None."""
    from src.backtest import _american_odds_to_prob
    if row_meta["odds_a"] is None or row_meta["odds_b"] is None:
        return None
    pa = _american_odds_to_prob(row_meta["odds_a"])
    pb = _american_odds_to_prob(row_meta["odds_b"])
    return pa / (pa + pb)


def evaluate(data_dir, test_size=1300, train_fn=None, quiet=False):
    """
    Trains on everything before the last `test_size` fights, evaluates on
    those. `train_fn(X, y, meta)` may be passed to customize training (e.g.
    recency weighting); it must return the fitted model list.
    Returns the overall (acc, log_loss, brier) on the test set.
    """
    from src.model import blend_market, ensemble_probs, fit_blend, train_models

    X, y, meta = generate_dataset(data_dir, with_meta=True)
    X_tr, y_tr, meta_tr = X[:-test_size], y[:-test_size], meta[:-test_size]
    X_te, y_te, meta_te = X[-test_size:], y[-test_size:], meta[-test_size:]

    blend = fit_blend(X_tr, y_tr, meta_tr)
    models = (train_fn or train_models)(X_tr, y_tr, meta_tr)

    probs = ensemble_probs(models, X_te)

    # Segment masks over the test set
    i_women = _col("is_women")
    i_class = _col("weight_class_ord")
    market = [_market_prob(m) for m in meta_te]

    segments = {
        "overall": [True] * len(y_te),
        "with betting line": [mp is not None for mp in market],
        "no betting line": [mp is None for mp in market],
        "women": [r[i_women] == 1.0 for r in X_te],
        "heavyweight": [r[i_class] == 9.0 for r in X_te],
        "debut (pre-UFC record)": [m["debut"] for m in meta_te],
    }

    lines = [f"test set: most recent {len(y_te)} fights "
             f"({meta_te[0]['date']:%Y-%m-%d} .. {meta_te[-1]['date']:%Y-%m-%d}), "
             f"trained on {len(y_tr)} earlier fights", "", "model:"]
    for label, mask in segments.items():
        sel = [(p, a) for p, a, keep in zip(probs, y_te, mask) if keep]
        m = _metrics([p for p, _ in sel], [a for _, a in sel]) if sel else None
        lines.append(_fmt(label, m, len(sel)))

    # Market baseline + two-stage blend, on fights with a line
    lined = [(mp, p, a) for mp, p, a in zip(market, probs, y_te) if mp is not None]
    if lined:
        mm = _metrics([mp for mp, _, _ in lined], [a for _, _, a in lined])
        lines.append("")
        lines.append("market baseline (devigged closing line):")
        lines.append(_fmt("with betting line", mm, len(lined)))
        if blend is not None:
            blended = [blend_market(p, mp, blend) for mp, p, _ in lined]
            bm = _metrics(blended, [a for _, _, a in lined])
            lines.append(f"model+market blend (weights {blend[0]:.2f} model, {blend[1]:.2f} market):")
            lines.append(_fmt("with betting line", bm, len(lined)))

            # What production actually does: blend when a line exists,
            # stats-only model otherwise.
            effective = [blend_market(p, mp, blend) if mp is not None else p
                         for mp, p in zip(market, probs)]
            em = _metrics(effective, list(y_te))
            lines.append("effective (blend where line exists, stats-only otherwise):")
            lines.append(_fmt("overall", em, len(y_te)))

    lines.extend(_evaluate_method(X_tr, y_tr, meta_tr, X_te, y_te, meta_te))

    overall = _metrics(list(probs), list(y_te))
    if not quiet:
        print("\n".join(lines))
    return overall


def _evaluate_method(X_tr, y_tr, meta_tr, X_te, y_te, meta_te):
    """
    Report lines for the method-of-victory head: 6-class log-loss and 3-class
    accuracy of the predicted fight-ending method (winner marginalized out),
    against a predict-the-most-common-method baseline.
    """
    from collections import Counter

    from src.model import METHOD_NAMES, fit_method_models, method_class, method_probs

    test = [(i, method_class(m["method"])) for i, m in enumerate(meta_te)]
    test = [(i, c) for i, c in test if c is not None]
    if not test:
        return []

    gb, _ = fit_method_models(X_tr, y_tr, meta_tr)
    probs = method_probs(gb, [X_te[i] for i, _ in test])

    actual_cls = [(0 if y_te[i] == 1 else 3) + c for i, c in test]
    ll6 = -sum(math.log(max(p[c], 1e-12)) for p, c in zip(probs, actual_cls)) / len(test)

    # 3-class view: which method ends the fight, whoever wins.
    method_p = [[p[m] + p[m + 3] for m in range(3)] for p in probs]
    acc = sum(max(range(3), key=mp.__getitem__) == c
              for mp, (_, c) in zip(method_p, test)) / len(test)
    ll3 = -sum(math.log(max(mp[c], 1e-12)) for mp, (_, c) in zip(method_p, test)) / len(test)

    counts = Counter(method_class(m["method"]) for m in meta_tr)
    counts.pop(None, None)
    base_rates = [counts[m] / sum(counts.values()) for m in range(3)]
    base_cls = max(range(3), key=base_rates.__getitem__)
    base_acc = sum(c == base_cls for _, c in test) / len(test)
    base_ll3 = -sum(math.log(max(base_rates[c], 1e-12)) for _, c in test) / len(test)

    return ["", "method of victory (KO/Sub/Dec, winner marginalized):",
            f"  {'model':<28s} acc {acc:6.1%}   log-loss {ll3:.4f}   "
            f"6-class log-loss {ll6:.4f}   (n={len(test)})",
            f"  {'baseline (train frequency)':<28s} acc {base_acc:6.1%}   log-loss {base_ll3:.4f}   "
            f"(always '{METHOD_NAMES[base_cls]}')"]


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1300
    evaluate(data_dir, test_size=size)
