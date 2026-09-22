"""
model.py

The trained prediction model, in two stages:

  1. A stats-only ensemble (logistic regression + random forest + gradient
     boosting) over leakage-free matchup differentials (see
     backtest.FEATURE_NAMES) — recency-weighted, with Bayesian-shrunk and
     opponent-adjusted rates. The betting market is NOT a feature.
  2. A market blend: when a betting line exists, the ensemble's probability
     is combined with the devigged market probability through a small
     logistic fit on a held-out calibration window.

Measured on a chronological backtest (train on the past, test on the most
recent ~1,300 fights; python3 -m src.evaluate):
  - effective system:  68.4% acc, 0.598 log-loss overall
  - blend, with line:  70.2% acc, 0.580 log-loss — better calibrated than
    the market itself (70.0%, 0.584)
  - old hand-tuned score: ~61% acc

Training (needs scikit-learn):   python3 -m src.model
Evaluation:                      python3 -m src.evaluate
Prediction:                      load_model() + predict_proba(...)

The ensemble is persisted with joblib to src/model.pkl. A pure-JSON logistic
fallback (src/model_weights.json) is also written, so prediction still works
without scikit-learn installed — just slightly less accurately.
"""
import json
import math
import os

from src.backtest import (
    FEATURE_NAMES, FEATURE_SIGNS, build_feature_row, build_current_states,
    _state_features, _add_time_features, _weightclass_context,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model_weights.json")


def _make_models():
    """Fresh, unfitted copies of the three ensemble members.

    Hyperparameters selected on a chronological validation split (see repo
    history): heavily regularized logit, deep-ish RF, shallow GB.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)
    gb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=7,
                                        l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=0.15, random_state=0)
    return [logit, rf, gb]


def fit_ensemble(X, y, sample_weight=None):
    """
    Fits the ensemble with symmetric A/B augmentation: mirror every fight
    (flip the differentials' signs, keep symmetric context features, flip the
    label) so the model can't learn anything from which fighter happened to
    be listed first. Returns the fitted model list.
    """
    import numpy as np

    X, y = np.array(X, dtype=float), np.array(y)
    signs = np.array(FEATURE_SIGNS)
    X_aug = np.vstack([X, X * signs])
    y_aug = np.concatenate([y, 1 - y])
    w_aug = None
    if sample_weight is not None:
        w = np.array(sample_weight, dtype=float)
        w_aug = np.concatenate([w, w])

    models = _make_models()
    for m in models:
        if w_aug is None:
            m.fit(X_aug, y_aug)
        else:
            # sklearn pipelines route fit params by step name
            if hasattr(m, "named_steps"):
                m.fit(X_aug, y_aug, logisticregression__sample_weight=w_aug)
            else:
                m.fit(X_aug, y_aug, sample_weight=w_aug)
    return models


# --- Method-of-victory model -------------------------------------------
# A second head over the same feature rows: 6 classes, A/B × KO/Sub/Dec.
# Mirroring a row (multiplying by FEATURE_SIGNS) swaps the fighters, which
# maps class c to (c + 3) % 6 — the same augmentation trick as the win model.
METHOD_NAMES = ["ko", "sub", "dec"]  # class = 3 * (0 if A won else 1) + index


def method_class(method):
    """Maps a METHOD string to 0 (KO/TKO), 1 (Sub), 2 (Decision), or None."""
    m = (method or "").strip()
    if "KO/TKO" in m or "Doctor's Stoppage" in m:
        return 0
    if "Submission" in m:
        return 1
    if "Decision" in m:
        return 2
    return None  # DQ, overturned, could-not-continue, ...


def _method_training_arrays(X, y, meta):
    """(X_aug, labels_aug, weights_aug) for the 6-class fit, mirrored."""
    import numpy as np

    weights = recency_weights(meta) or [1.0] * len(meta)
    rows, labels, w = [], [], []
    for row, outcome, m, wt in zip(X, y, meta, weights):
        c = method_class(m.get("method"))
        if c is None:
            continue
        rows.append(row)
        labels.append((0 if outcome == 1 else 3) + c)
        w.append(wt)

    X_ = np.array(rows, dtype=float)
    L = np.array(labels)
    W = np.array(w, dtype=float)
    signs = np.array(FEATURE_SIGNS)
    return np.vstack([X_, X_ * signs]), np.concatenate([L, (L + 3) % 6]), np.concatenate([W, W])


def fit_method_models(X, y, meta):
    """
    Fits the 6-class method head: a gradient-boosted model (primary) and a
    multinomial logit (JSON fallback). Returns (gb, logit_pipeline).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_aug, L_aug, W_aug = _method_training_arrays(X, y, meta)

    gb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=7,
                                        l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=0.15, random_state=0)
    gb.fit(X_aug, L_aug, sample_weight=W_aug)

    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
    logit.fit(X_aug, L_aug, logisticregression__sample_weight=W_aug)
    return gb, logit


def _method_json_payload(logit):
    """Multinomial logit folded into plain coefficient lists (scaler included)."""
    import numpy as np

    scaler = logit.named_steps["standardscaler"]
    clf = logit.named_steps["logisticregression"]
    W = clf.coef_ / scaler.scale_
    b = clf.intercept_ - W @ scaler.mean_
    return {"classes": [int(c) for c in clf.classes_],
            "weights": [[float(v) for v in row] for row in W],
            "bias": [float(v) for v in b]}


def _method_probs_one(model, row):
    """The 6 class probabilities for one feature row, under either model kind."""
    if model["kind"] == "logit":
        payload = model["method"]
        zs = [sum(w * x for w, x in zip(ws, row)) + b
              for ws, b in zip(payload["weights"], payload["bias"])]
        mx = max(zs)
        exps = [math.exp(z - mx) for z in zs]
        total = sum(exps)
        by_class = {c: e / total for c, e in zip(payload["classes"], exps)}
    else:
        clf = model["method_model"]
        probs = clf.predict_proba([row])[0]
        by_class = {int(c): float(p) for c, p in zip(clf.classes_, probs)}
    return [by_class.get(c, 0.0) for c in range(6)]


def method_probs(clf, X):
    """6-class probabilities for many rows at once, symmetrized over orderings."""
    import numpy as np

    X = np.array(X, dtype=float)
    signs = np.array(FEATURE_SIGNS)

    def full(P, classes):
        out = np.zeros((len(P), 6))
        for i, c in enumerate(classes):
            out[:, int(c)] = P[:, i]
        return out

    p_ab = full(clf.predict_proba(X), clf.classes_)
    p_ba = full(clf.predict_proba(X * signs), clf.classes_)
    perm = [3, 4, 5, 0, 1, 2]  # mirrored ordering swaps the A and B classes
    return (p_ab + p_ba[:, perm]) / 2


def predict_method_proba(model, states, physical, pool_means, name_a, name_b):
    """
    Method-of-victory probabilities for a matchup, or None if the loaded
    model has no method head. Returns {"a": {"ko": p, "sub": p, "dec": p},
    "b": {...}} — six absolute probabilities summing to 1 ("a"/"ko" is
    P(name_a wins by KO)). Symmetrized over both orderings.
    """
    has_head = model.get("method_model") if model["kind"] == "ensemble" else model.get("method")
    if not has_head:
        return None

    row_ab, row_ba = _matchup_rows(states, physical, pool_means, name_a, name_b)
    p_ab = _method_probs_one(model, row_ab)
    p_ba = _method_probs_one(model, row_ba)
    p = [(p_ab[c] + p_ba[(c + 3) % 6]) / 2 for c in range(6)]
    return {"a": dict(zip(METHOD_NAMES, p[:3])), "b": dict(zip(METHOD_NAMES, p[3:]))}


# A fight this many years before the newest one counts half as much during
# training: the sport evolves, so recent fights should shape the fit more.
# Gentle on purpose — old fights still carry most of the sample size.
TRAIN_RECENCY_HALF_LIFE_YEARS = 12.0


def recency_weights(meta):
    """Per-sample training weights decaying with fight age. None if undated."""
    dated = [m["date"] for m in meta if m["date"] is not None]
    if not dated:
        return None
    latest = max(dated)
    return [0.5 ** (((latest - m["date"]).days / 365.25) / TRAIN_RECENCY_HALF_LIFE_YEARS)
            if m["date"] is not None else 0.5 for m in meta]


def train_models(X, y, meta):
    """The standard training recipe: recency-weighted ensemble fit."""
    return fit_ensemble(X, y, sample_weight=recency_weights(meta))


def train(data_dir, model_path=MODEL_PATH, weights_path=WEIGHTS_PATH):
    """
    Trains the ensemble on every fight in the dataset and persists both the
    ensemble (joblib) and the logistic fallback (JSON).
    """
    import joblib
    import numpy as np

    from src.backtest import generate_dataset

    X, y, meta = generate_dataset(data_dir, with_meta=True)
    blend = fit_blend(X, y, meta)
    models = train_models(X, y, meta)
    logit = models[0]
    method_gb, method_logit = fit_method_models(X, y, meta)

    joblib.dump({"features": FEATURE_NAMES, "models": models, "blend": blend,
                 "method_model": method_gb,
                 "training_fights": int(len(X))}, model_path, compress=3)

    # JSON fallback: the logit alone, scaler folded into the coefficients so
    # prediction is possible with no ML dependencies at all.
    scaler = logit.named_steps["standardscaler"]
    clf = logit.named_steps["logisticregression"]
    w = clf.coef_[0] / scaler.scale_
    b = clf.intercept_[0] - float(np.dot(clf.coef_[0], scaler.mean_ / scaler.scale_))
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURE_NAMES, "weights": [float(v) for v in w],
                   "bias": float(b), "blend": blend,
                   "method": _method_json_payload(method_logit),
                   "training_fights": int(len(X))}, f, indent=2)

    return len(X)


def load_model(model_path=MODEL_PATH, weights_path=WEIGHTS_PATH):
    """
    Returns the best available trained model, or None if none exists (or the
    feature set has changed since training, which would misalign columns).
    Prefers the pickled ensemble; falls back to the JSON logistic weights.
    """
    if os.path.exists(model_path):
        try:
            import joblib
            payload = joblib.load(model_path)
            if payload.get("features") == FEATURE_NAMES:
                return {"kind": "ensemble", **payload}
        except ImportError:
            pass  # scikit-learn/joblib not installed — fall through to JSON

    if os.path.exists(weights_path):
        with open(weights_path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("features") == FEATURE_NAMES:
            return {"kind": "logit", **payload}

    return None


def ensemble_probs(models, X):
    """P(A wins) for many rows at once, symmetrized over both orderings."""
    import numpy as np

    X = np.array(X, dtype=float)
    signs = np.array(FEATURE_SIGNS)
    p_ab = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    p_ba = np.mean([m.predict_proba(X * signs)[:, 1] for m in models], axis=0)
    return (p_ab + (1 - p_ba)) / 2


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# The last N training fights are held out (chronologically) to calibrate the
# market blend on out-of-sample model probabilities — calibrating in-sample
# would overweight the model, since the RF's training-set probabilities are
# near-perfect.
BLEND_CALIBRATION_FIGHTS = 800


def fit_blend(X, y, meta):
    """
    Fits the second stage: how to combine the stats model's probability with
    the betting market's, as a logistic over the two logits. Trained on the
    calibration tail's fights that had a line. No intercept, so the blend
    keeps predict(a, b) == 1 - predict(b, a). Returns [w_model, w_market],
    or None when there aren't enough lined fights to calibrate on.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from src.backtest import _american_odds_to_prob

    if len(y) <= BLEND_CALIBRATION_FIGHTS * 2:
        return None
    cut = len(y) - BLEND_CALIBRATION_FIGHTS
    models = train_models(X[:cut], y[:cut], meta[:cut])
    probs = ensemble_probs(models, X[cut:])

    rows, labels = [], []
    for p, outcome, m in zip(probs, y[cut:], meta[cut:]):
        if m["odds_a"] is None or m["odds_b"] is None:
            continue
        pa, pb = _american_odds_to_prob(m["odds_a"]), _american_odds_to_prob(m["odds_b"])
        rows.append([_logit(p), _logit(pa / (pa + pb))])
        labels.append(outcome)
    if len(labels) < 200:
        return None

    R, L = np.array(rows), np.array(labels)
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(np.vstack([R, -R]), np.concatenate([L, 1 - L]))  # mirrored for symmetry
    return [float(clf.coef_[0][0]), float(clf.coef_[0][1])]


def blend_market(p_model, p_market, blend):
    """Final probability from the stats model's and the market's, via the blend."""
    z = blend[0] * _logit(p_model) + blend[1] * _logit(p_market)
    return 1 / (1 + math.exp(-z))


def _proba_one(model, row):
    """P(A wins) for one feature row, under either model kind."""
    if model["kind"] == "logit":
        z = sum(w * x for w, x in zip(model["weights"], row)) + model["bias"]
        return 1 / (1 + math.exp(-z))
    probs = [m.predict_proba([row])[0][1] for m in model["models"]]
    return sum(probs) / len(probs)


def _matchup_rows(states, physical, pool_means, name_a, name_b):
    """
    The (row_ab, row_ba) feature vectors for a hypothetical matchup today —
    the same fight seen from each corner, for order-invariant prediction.
    """
    from datetime import datetime
    today = datetime.now()

    phys_a, phys_b = physical[name_a], physical[name_b]
    fa = _add_time_features(_state_features(states[name_a], pool_means), states[name_a], phys_a, today, pool_means)
    fb = _add_time_features(_state_features(states[name_b], pool_means), states[name_b], phys_b, today, pool_means)

    # Fight-level context, inferred: assume the bout happens in the HEAVIER
    # of the two fighters' most recent weight classes — when someone moves
    # divisions to make a fight, they almost always move up to meet the
    # bigger fighter. That also makes the mover's class_shift positive, so
    # the model sees that their stats were earned against smaller opposition.
    from src.backtest import _class_shift, _matchup_history_features
    wc_a = states[name_a]["last_class"]
    wc_b = states[name_b]["last_class"]
    ord_a, _ = _weightclass_context(wc_a)
    ord_b, _ = _weightclass_context(wc_b)
    wc = wc_a if ord_a >= ord_b else wc_b
    wc = wc or wc_a or wc_b
    wc_ord, is_women = _weightclass_context(wc)
    shift_a = _class_shift(wc_ord, wc_a)
    shift_b = _class_shift(wc_ord, wc_b)
    h2h_edge, co_edge, co_count = _matchup_history_features(
        states[name_a], states[name_b], name_a, name_b)
    shared = {"weight_class_ord": wc_ord, "is_women": is_women, "common_opp_count": co_count}
    # signed-from-A's-perspective context flips/swaps for the B-first ordering
    context_ab = {**shared, "h2h_edge": h2h_edge, "common_opp_edge": co_edge,
                  "class_shift_a": shift_a, "class_shift_b": shift_b}
    context_ba = {**shared, "h2h_edge": -h2h_edge, "common_opp_edge": -co_edge,
                  "class_shift_a": shift_b, "class_shift_b": shift_a}

    row_ab = build_feature_row(fa, fb, phys_a, phys_b, context_ab)
    row_ba = build_feature_row(fb, fa, phys_b, phys_a, context_ba)
    return row_ab, row_ba


def predict_proba(model, states, physical, pool_means, name_a, name_b,
                  odds_prob_diff=0.0):
    """
    Returns P(name_a beats name_b) using the trained model.
    Symmetrized over both orderings so predict(a, b) == 1 - predict(b, a).

    odds_prob_diff is the market's devigged P(a) − P(b) when a live betting
    line is known (see src.value); 0 (neutral) for hypothetical matchups.
    The line is no longer a model feature — when given, it's combined with
    the stats-only probability through the trained two-stage blend.
    """
    row_ab, row_ba = _matchup_rows(states, physical, pool_means, name_a, name_b)

    p_ab = _proba_one(model, row_ab)
    p_ba = _proba_one(model, row_ba)
    p = (p_ab + (1 - p_ba)) / 2

    if odds_prob_diff and model.get("blend"):
        p = blend_market(p, (1 + odds_prob_diff) / 2, model["blend"])
    return p


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    n = train(data_dir)
    print(f"trained ensemble on {n} fights -> {MODEL_PATH} (+ JSON fallback)")
