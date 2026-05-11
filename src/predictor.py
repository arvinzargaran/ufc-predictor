"""
predictor.py

Given two fighter names and a dictionary of Fighter objects,
compare their statistics and predict a winner with a confidence score.

"""
from openpyxl.styles.builtins import total

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

def compute_score(fighter: Fighter) -> float:
    """
    Compute a single numeric score for a fighter based on their
    weighted stats and physical attributes.
    Returns a raw float — not yet normalized to 0-100.
    """
    # --- Win rate ---
    # Calculate from raw record. Guard against division by zero
    # in case a fighter has never had a professional bout.
    total_fights = fighter.wins + fighter.losses + fighter.draws
    win_rate = fighter.wins / total_fights if total_fights > 0 else 0.0

    # --- Stat score ---
    # Loop over every stat in STAT_WEIGHTS and accumulate a weighted total.
    stat_score = 0.0
    for stat, weight in STAT_WEIGHTS.items():

        # getattr(object, name) fetches fighter.slpm, fighter.str_acc, etc.
        # using the string key from the dictionary.
        value = getattr(fighter, stat)

        if stat == "sapm":
            # sapm is the only stat where LOWER is better.
            # We invert it by subtracting from a fixed ceiling (10.0).
            # A fighter absorbing 2.0 strikes/min scores 8.0 here.
            # A fighter absorbing 8.0 strikes/min scores only 2.0.
            value = 10.0 - value

        stat_score += value * weight

    # --- Attribute score ---
    # Same pattern as stat_score but for physical attributes.
    attribute_score = 0.0
    attribute_score += win_rate          * ATTRIBUTE_WEIGHTS["win_rate"]
    attribute_score += fighter.reach     * ATTRIBUTE_WEIGHTS["reach"]
    attribute_score += fighter.height    * ATTRIBUTE_WEIGHTS["height"]
    attribute_score += fighter.weight    * ATTRIBUTE_WEIGHTS["weight"]

    # --- Combined score ---
    return stat_score + attribute_score


def predict(name1: str, name2: str, fighters: dict[str, Fighter]) -> dict:
    """
    predict the winner of a fight
    returns a dictionary with the predicted winner, loser, and a confidence score from 0-100

    """

    # --- look up both fighters ---
    fighter1 = get_fighter(name1, fighters)
    fighter2 = get_fighter(name2, fighters)

    # --- compute raw score ---
    score1 = compute_score(fighter1)
    score2 = compute_score(fighter2)

    # --- Stance bonus ---
    # Southpaw fighters have a documented edge against orthodox opponents.
    # We apply a small multiplier to the southpaw fighter's score only
    # when the matchup is explicitly southpaw vs orthodox.
    STANCE_BONUS = 1.05  # 5% bonus — meaningful but not decisive

    f1_stance = (fighter1.stance or "").lower()
    f2_stance = (fighter2.stance or "").lower()

    if f1_stance == "southpaw" and f2_stance == "orthodox":
        score1 *= STANCE_BONUS
    elif f2_stance == "southpaw" and f1_stance == "orthodox":
        score2 *= STANCE_BONUS

    # --- Normalize edge case ----
    # we express each fighter's score as a share of combied total
    # Example: scores of 8.0 and 12.0 -> fighter2 wins with 60% confidence

    total = score1 + score2
    if total == 0:
        confidence1 = 50.0
        confidence2 = 50.0
    else:
        confidence1 = (score1 / total) * 100
        confidence2 = (score2 / total) * 100

    # --- Determine winner ---

    if confidence1 > confidence2:

        winner = fighter1
        loser = fighter2
        confidence = confidence1

    else:
        winner = fighter2
        loser = fighter1
        confidence = confidence2

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
    }










