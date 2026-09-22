"""
predictor.py

Given two fighter names and a dictionary of Fighter objects,
compare their statistics and predict a winner with a confidence score.

"""
import math

from src.fighter import Fighter


# Performance stats — how the fighter actually performs inside the octagon.
# Weights reflect how much each stat influences the predicted outcome.
STAT_WEIGHTS = {
    "slpm":    0.15,   # Significant strikes landed per minute — offensive output
    "str_acc": 0.10,   # Strike accuracy — how clean the striking is
    "sapm":    0.10,   # Strikes absorbed per minute — LOWER is better
    "str_def": 0.10,   # Strike defence — ability to avoid being hit
    "td_avg":  0.10,   # Takedown average — grappling offence
    "td_acc":  0.05,   # Takedown accuracy — precision of takedown attempts
    "td_def":  0.10,   # Takedown defence — stops opponent's takedowns
    "sub_avg": 0.05,   # Submission average — finishing threat on the ground
}

# Physical attributes and record — contextual advantages outside of raw stats.
ATTRIBUTE_WEIGHTS = {
    "win_rate": 0.15,  # Wins / total fights — proven track record
    "reach":    0.05,  # Reach in inches — striking range advantage
    "height":   0.03,  # Height in inches — leverage and kicking range
    "weight":   0.02,  # Weight in lbs — natural size advantage
}

# Total across both dicts = 1.00
# STAT_WEIGHTS:      0.15+0.10+0.10+0.10+0.10+0.05+0.10+0.05 = 0.75
# ATTRIBUTE_WEIGHTS: 0.15+0.05+0.03+0.02 = 0.25

# A small, fixed edge for a southpaw fighting an orthodox opponent, expressed
# in the same units as a z-score-weighted stat contribution (roughly
# equivalent to being ~1 standard deviation above average on a 0.15-weight stat).
STANCE_BONUS = 0.15

# Upper weight (lbs) of each UFC weight class, used only to flag cross-class
# matchups — the stats/attributes above aren't meaningful comparisons across
# wildly different weight classes.
WEIGHT_CLASS_CUTOFFS = [125, 135, 145, 155, 170, 185, 205, 265]


def get_fighter(name: str, fighters: dict[str, Fighter]) -> Fighter:

    """
    look up a fighter by full name in the fighters dictionary.
    raises a clear error if the same isn't found.

    """
    #try exact name match.

    if name in fighters:
        return fighters[name]

    #if the exact match fails, try case insensitive.

    name_lower = name.lower()
    for key in fighters:
        if key.lower() == name_lower:
            return fighters[key]

    #if we still haven't found them raise an error.

    raise ValueError(f"fighter {name} not found in the database")


def _weight_class_index(weight: float) -> int:
    for i, cutoff in enumerate(WEIGHT_CLASS_CUTOFFS):
        if weight <= cutoff:
            return i
    return len(WEIGHT_CLASS_CUTOFFS)


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance ** 0.5


def _zscore(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std


def _compute_norm_stats(fighters: dict[str, Fighter]) -> dict:
    """
    Computes (mean, std) for every stat/attribute across the whole fighter
    pool, so raw values (on wildly different scales — lbs, inches, per-minute
    rates, percentages) can be compared fairly via z-scores instead of the
    weights being swamped by whichever field happens to have the largest units.
    """
    norm_stats = {}

    for field in STAT_WEIGHTS:
        values = [getattr(f, field) for f in fighters.values()]
        norm_stats[field] = _mean_std(values)

    for field in ("reach", "height", "weight"):
        values = [getattr(f, field) for f in fighters.values()]
        norm_stats[field] = _mean_std(values)

    # win_rate is only meaningful for fighters with a recorded fight history;
    # fighters with none are treated as "average" (neutral) rather than 0.
    win_rates = [f.win_rate() for f in fighters.values() if (f.wins + f.losses + f.draws) > 0]
    norm_stats["win_rate"] = _mean_std(win_rates) if win_rates else (0.5, 1.0)

    return norm_stats


def compute_score(fighter: Fighter, norm_stats: dict) -> float:
    """
    Compute a single numeric score for a fighter based on their
    weighted, z-score-normalized stats and physical attributes.
    """
    total_fights = fighter.wins + fighter.losses + fighter.draws
    if total_fights > 0:
        win_rate = fighter.win_rate()
    else:
        # No recorded fights — treat as an average fighter rather than
        # penalizing them with a 0.0 win rate they haven't actually earned.
        win_rate = norm_stats["win_rate"][0]

    # --- Stat score ---
    stat_score = 0.0
    for stat, weight in STAT_WEIGHTS.items():
        value = getattr(fighter, stat)
        mean, std = norm_stats[stat]
        z = _zscore(value, mean, std)

        if stat == "sapm":
            # sapm is the only stat where LOWER is better, so its z-score
            # contribution is inverted.
            z = -z

        stat_score += z * weight

    # --- Attribute score ---
    attribute_score = 0.0
    attribute_score += _zscore(win_rate, *norm_stats["win_rate"]) * ATTRIBUTE_WEIGHTS["win_rate"]
    attribute_score += _zscore(fighter.reach, *norm_stats["reach"]) * ATTRIBUTE_WEIGHTS["reach"]
    attribute_score += _zscore(fighter.height, *norm_stats["height"]) * ATTRIBUTE_WEIGHTS["height"]
    attribute_score += _zscore(fighter.weight, *norm_stats["weight"]) * ATTRIBUTE_WEIGHTS["weight"]

    return stat_score + attribute_score


def _predict_ml(fighter1: Fighter, fighter2: Fighter, data_dir: str) -> dict | None:
    """
    Predicts using the trained logistic-regression model.
    Returns None if no trained model is available (caller falls back to the
    hand-tuned score) or if either fighter has no UFC fight history yet
    (the model was trained only on fighters with prior fights).
    """
    # Imported lazily: backtest imports from this module at load time, so a
    # top-level import here would be circular.
    from src.model import load_model, predict_proba, predict_method_proba
    from src.backtest import build_current_states

    model = load_model()
    if model is None:
        return None

    name1 = fighter1.full_name.strip().lower()
    name2 = fighter2.full_name.strip().lower()

    states, physical, pool_means = build_current_states(data_dir)
    if name1 not in physical or name2 not in physical:
        return None

    for name in (name1, name2):
        s = states[name]
        if s["wins"] + s["losses"] + s["draws"] == 0:
            # UFC debut — predictable only if we know their pre-UFC record
            pre = physical[name]
            if pre.get("pre_ufc_wins", 0) + pre.get("pre_ufc_losses", 0) == 0:
                return None

    p1 = predict_proba(model, states, physical, pool_means, name1, name2)

    winner, loser = (fighter1, fighter2) if p1 >= 0.5 else (fighter2, fighter1)
    confidence = max(p1, 1 - p1) * 100

    same_weight_class = _weight_class_index(fighter1.weight) == _weight_class_index(fighter2.weight)

    # How each fighter wins, IF they win: the method head's absolute
    # probabilities for that fighter, renormalized over their three outcomes.
    methods = predict_method_proba(model, states, physical, pool_means, name1, name2)
    conditional_methods = None
    if methods is not None:
        conditional_methods = {}
        for key, fighter in (("a", fighter1), ("b", fighter2)):
            total = sum(methods[key].values())
            conditional_methods[fighter.full_name] = (
                {m: p / total for m, p in methods[key].items()} if total else None)

    return {
        "winner": winner,
        "loser": loser,
        "confidence": round(confidence, 1),
        "score1": round(p1 * 100, 4),
        "score2": round((1 - p1) * 100, 4),
        "fighter1": fighter1,
        "fighter2": fighter2,
        "same_weight_class": same_weight_class,
        "methods": conditional_methods,
    }


def predict(name1: str, name2: str, fighters: dict[str, Fighter], data_dir: str = None) -> dict:
    """
    predict the winner of a fight
    returns a dictionary with the predicted winner, loser, and a confidence score from 0-100

    If data_dir is given and a trained model exists (src/model.pkl or
    src/model_weights.json, produced by `python3 -m src.model`), uses the
    trained model — ~68% accuracy on a chronological backtest, vs ~61% for
    the hand-tuned score below, which remains the fallback.
    """

    # --- look up both fighters ---
    fighter1 = get_fighter(name1, fighters)
    fighter2 = get_fighter(name2, fighters)

    if data_dir is not None:
        ml_result = _predict_ml(fighter1, fighter2, data_dir)
        if ml_result is not None:
            return ml_result

    # --- compute normalized score ---
    norm_stats = _compute_norm_stats(fighters)
    score1 = compute_score(fighter1, norm_stats)
    score2 = compute_score(fighter2, norm_stats)

    # --- Stance bonus ---
    # Southpaw fighters have a documented edge against orthodox opponents.
    # Applied as a fixed additive bonus (scores are z-score-weighted sums,
    # which can be small or negative, so a multiplier would be unreliable).
    f1_stance = (fighter1.stance or "").lower()
    f2_stance = (fighter2.stance or "").lower()

    if f1_stance == "southpaw" and f2_stance == "orthodox":
        score1 += STANCE_BONUS
    elif f2_stance == "southpaw" and f1_stance == "orthodox":
        score2 += STANCE_BONUS

    # --- Confidence ---
    # Map the score difference through a sigmoid so confidence is always
    # bounded in (0, 100) and a 0 difference is exactly 50/50 — this also
    # handles negative or zero scores correctly, unlike a plain score-share ratio.
    diff = score1 - score2
    confidence1 = 100 / (1 + math.exp(-diff))
    confidence2 = 100 - confidence1

    # --- Determine winner ---
    if confidence1 > confidence2:
        winner = fighter1
        loser = fighter2
        confidence = confidence1
    else:
        winner = fighter2
        loser = fighter1
        confidence = confidence2

    same_weight_class = _weight_class_index(fighter1.weight) == _weight_class_index(fighter2.weight)

    # --- return structured result ---
    # a dictionary lets display.py extract exactly what it needs

    return {
        "winner": winner,
        "loser": loser,
        "confidence": round(confidence, 1),
        "score1": round(score1, 4),
        "score2": round(score2, 4),
        "fighter1": fighter1,
        "fighter2": fighter2,
        "same_weight_class": same_weight_class,
    }
