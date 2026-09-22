"""
backtest.py

Measures the predictor's real accuracy against UFC history.

The naive approach — load_fighters() aggregates a fighter's whole career,
then predictor.predict() is asked to "predict" a fight already baked into
those averages — is data leakage: the model has already seen the answer.

This module instead replays every fight in chronological order. Before each
fight, it snapshots each fighter's record/stats from *only their fights so
far*, asks the predictor to pick a winner using that snapshot, then updates
the running totals with the real result before moving to the next fight.
"""
import csv
from collections import defaultdict
from datetime import datetime

from src.fighter import Fighter
from src.loader import load_physical_profiles, _parse_of, _parse_int
from src.predictor import compute_score, _compute_norm_stats, STANCE_BONUS


def _parse_date(val):
    """
    Parses dates like "Jul 19, 1987" (fighter DOBs, abbreviated month) and
    "April 25, 2026" (event dates, full month).
    """
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(val.strip(), fmt)
        except (ValueError, AttributeError, TypeError):
            continue
    return None


def _load_event_dates(data_dir):
    """Returns {event_name: datetime}. Empty if the events CSV is absent."""
    dates = {}
    try:
        with open(f"{data_dir}/ufc_event_details.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                date = _parse_date(row["DATE"])
                if date:
                    dates[row["EVENT"].strip()] = date
    except FileNotFoundError:
        pass
    return dates


def _american_odds_to_prob(odds):
    """Converts American odds (e.g. -130, +102) to implied win probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def _load_odds(data_dir):
    """
    Returns {(date_str, frozenset({name_a, name_b})):
             {"prob": {name: devigged_prob}, "american": {name: odds}}}.
    Empty if data/ufc_odds.csv is absent (odds are optional — the feature
    falls back to neutral 0 for fights without a betting line).
    """
    odds = {}
    try:
        with open(f"{data_dir}/ufc_odds.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                red = row["RED"].strip().lower()
                blue = row["BLUE"].strip().lower()
                try:
                    o_red, o_blue = float(row["RED_ODDS"]), float(row["BLUE_ODDS"])
                except ValueError:
                    continue
                p_red = _american_odds_to_prob(o_red)
                p_blue = _american_odds_to_prob(o_blue)
                total = p_red + p_blue  # devig: raw probs include the book's margin
                key = (row["DATE"], frozenset([red, blue]))
                odds[key] = {
                    "red": red,
                    "prob": {red: p_red / total, blue: p_blue / total},
                    "american": {red: o_red, blue: o_blue},
                    "rank": {red: _rank_score(row.get("RED_RANK")),
                             blue: _rank_score(row.get("BLUE_RANK"))},
                    "empty_arena": 1.0 if row.get("EMPTY_ARENA") == "1.0" else 0.0,
                }
    except FileNotFoundError:
        pass
    return odds


def _find_odds(odds, date, name_a, name_b):
    """The odds entry for a fight, or None if no line (or names missing from it)."""
    if date is None:
        return None
    line = odds.get((date.strftime("%Y-%m-%d"), frozenset([name_a, name_b])))
    if line is None or name_a not in line["prob"] or name_b not in line["prob"]:
        return None
    return line


def _attach_pre_ufc_records(physical, data_dir):
    """
    Attaches each fighter's estimated pre-UFC record to their physical profile
    (pre_ufc_wins/losses/draws), computed as overall career MMA record (from
    data/ufc_fighter_records.csv, scraped from ufc.com) minus their UFC record
    (counted from our results CSV). Approximation: fights taken outside the
    UFC *after* a UFC stint get counted as "pre-UFC" too — rare, accepted.
    Fighters missing from the records file get zeros. No-op if file absent.
    """
    overall = {}
    try:
        with open(f"{data_dir}/ufc_fighter_records.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                overall[row["NAME"].strip().lower()] = (
                    _parse_int(row["WINS"]), _parse_int(row["LOSSES"]), _parse_int(row["DRAWS"]))
    except FileNotFoundError:
        pass

    ufc_totals = defaultdict(lambda: [0, 0, 0])
    with open(f"{data_dir}/ufc_fight_results.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bout, outcome = row["BOUT"], row["OUTCOME"].strip()
            if " vs. " not in bout or "/" not in outcome:
                continue
            a, b = (n.strip().lower() for n in bout.split(" vs. "))
            res = outcome.split("/")[0]
            if res == "W":
                ufc_totals[a][0] += 1
                ufc_totals[b][1] += 1
            elif res == "L":
                ufc_totals[a][1] += 1
                ufc_totals[b][0] += 1

    for name, phys in physical.items():
        ov = overall.get(name)
        ufc = ufc_totals.get(name, [0, 0, 0])
        if ov:
            phys["pre_ufc_wins"] = max(ov[0] - ufc[0], 0)
            phys["pre_ufc_losses"] = max(ov[1] - ufc[1], 0)
            phys["pre_ufc_draws"] = max(ov[2], 0)
        else:
            phys["pre_ufc_wins"] = phys["pre_ufc_losses"] = phys["pre_ufc_draws"] = 0


def _pre_ufc_features(phys):
    """(experience, win_rate) from the pre-UFC record; neutral 0.5 rate if none."""
    total = phys["pre_ufc_wins"] + phys["pre_ufc_losses"] + phys["pre_ufc_draws"]
    rate = phys["pre_ufc_wins"] / total if total else 0.5
    return total, rate


def _load_fights_chronological(data_dir):
    """
    Joins ufc_fight_results.csv and ufc_fight_stats.csv on (EVENT, BOUT) and
    returns a list of fights in oldest-to-newest order (the source CSVs are
    newest-first, so we reverse them).
    """
    stats_by_fight = defaultdict(list)
    with open(f"{data_dir}/ufc_fight_stats.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["EVENT"].strip(), row["BOUT"].strip())
            stats_by_fight[key].append(row)

    event_dates = _load_event_dates(data_dir)

    fights = []
    with open(f"{data_dir}/ufc_fight_results.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bout = row["BOUT"].strip()
            outcome = row["OUTCOME"].strip()

            if " vs. " not in bout or "/" not in outcome:
                continue  # skip no-contests / malformed rows

            fighter_a, fighter_b = (n.strip().lower() for n in bout.split(" vs. "))
            a_result, b_result = outcome.split("/")

            event = row["EVENT"].strip()
            key = (event, bout)
            fights.append({
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "a_result": a_result,
                "b_result": b_result,
                "method": row["METHOD"].strip(),
                "end_round": _parse_int(row.get("ROUND", "")),
                "date": event_dates.get(event),
                "five_round": "5 Rnd" in row.get("TIME FORMAT", ""),
                "weightclass": row.get("WEIGHTCLASS", "").strip(),
                "rounds": stats_by_fight.get(key, []),
            })

    fights.reverse()  # oldest first (source CSV is newest-first)
    # Refine with real dates where available (stable sort keeps same-day order).
    fights.sort(key=lambda f: f["date"] or datetime.min)
    return fights


def _new_state():
    return {
        "wins": 0, "losses": 0, "draws": 0,
        "str_landed": 0, "str_attempted": 0,
        "str_absorbed": 0, "opp_str_attempted": 0,
        "td_landed": 0, "td_attempted": 0,
        "opp_td_landed": 0, "opp_td_attempted": 0,
        "sub_att": 0, "rounds": 0,
        # extra signals tracked for the ML feature set
        "kd": 0, "kd_against": 0,
        "ctrl_seconds": 0, "opp_ctrl_seconds": 0,
        "streak": 0,  # current win streak (resets to 0 on a loss/draw)
        "elo": 1500.0,  # opponent-quality-adjusted skill rating
        "elo_log": [],  # own pre-fight Elo, newest last (for the trend feature)
        "finish_wins": 0,   # wins by KO/TKO or submission
        "ko_losses": 0,     # times knocked out — durability signal
        "recent": [],       # 1/0 results of most recent fights, newest last
        "opp_results": {},  # opponent name -> list of results vs them (1/0/0.5)
        "fight_log": [],    # per-fight stat totals, newest last (for recent-window rates)
        "first_date": None, "last_date": None,  # for career length / layoff
        "five_round_fights": 0,   # championship/main-event experience
        "last_class": None,       # weight class of previous fight
        "changed_class": False,   # did the last fight involve a class switch?
        "opp_elos": [],           # opponents' pre-fight Elo, newest last
        # Glicko-2 rating (internal scale): rating, deviation, volatility.
        # Deviation widens with inactivity — a returning fighter is uncertain.
        "g_mu": 0.0, "g_phi": 350 / 173.7178, "g_sigma": 0.06,
        "g_last": None,  # date of last rated fight, for inactivity widening
        "r1_finishes": 0, "late_finishes": 0,  # fast starter vs late closer
        # per-round striking output, to expose cardio (round 1 vs round 3+)
        "r1_landed": 0, "r1_rounds": 0,
        "r3p_landed": 0, "r3p_rounds": 0,
        # where strikes happen (style: distance striker vs clinch/ground)
        "distance_att": 0, "clinch_att": 0, "ground_att": 0,
        # strike targets (leg kicker, body worker)
        "head_landed": 0, "body_landed": 0, "leg_landed": 0,
    }


ELO_K = 32  # standard Elo update speed


def _update_elo(state_winner, state_loser):
    expected = 1 / (1 + 10 ** ((state_loser["elo"] - state_winner["elo"]) / 400))
    delta = ELO_K * (1 - expected)
    state_winner["elo"] += delta
    state_loser["elo"] -= delta


# --- Glicko-2 (per Glickman's spec, τ=0.5, one game per rating period) ---
GLICKO_TAU = 0.5
GLICKO_MAX_PHI = 350 / 173.7178


def _glicko_widen(state, date):
    """Widens rating deviation for inactivity (one 'period' per quarter-year)."""
    if state["g_last"] is not None and date is not None:
        periods = max((date - state["g_last"]).days, 0) / 91.3
        state["g_phi"] = min((state["g_phi"] ** 2 + state["g_sigma"] ** 2 * periods) ** 0.5,
                             GLICKO_MAX_PHI)


def _glicko_update(state_winner, state_loser, date):
    import math

    for state in (state_winner, state_loser):
        _glicko_widen(state, date)

    pre = [(s["g_mu"], s["g_phi"], s["g_sigma"]) for s in (state_winner, state_loser)]

    for state, score, (mu, phi, sigma), (mu_j, phi_j, _) in (
        (state_winner, 1.0, pre[0], pre[1]),
        (state_loser, 0.0, pre[1], pre[0]),
    ):
        g = 1 / math.sqrt(1 + 3 * phi_j ** 2 / math.pi ** 2)
        E = 1 / (1 + math.exp(-g * (mu - mu_j)))
        v = 1 / (g ** 2 * E * (1 - E))
        delta = v * g * (score - E)

        # volatility update (Illinois algorithm from the spec)
        a = math.log(sigma ** 2)
        def f(x):
            ex = math.exp(x)
            return (ex * (delta ** 2 - phi ** 2 - v - ex) / (2 * (phi ** 2 + v + ex) ** 2)
                    - (x - a) / GLICKO_TAU ** 2)
        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * GLICKO_TAU) < 0:
                k += 1
            B = a - k * GLICKO_TAU
        fa_, fb_ = f(A), f(B)
        while abs(B - A) > 1e-6:
            C = A + (A - B) * fa_ / (fb_ - fa_)
            fc_ = f(C)
            if fc_ * fb_ <= 0:
                A, fa_ = B, fb_
            else:
                fa_ /= 2
            B, fb_ = C, fc_
        sigma_new = math.exp(A / 2)

        phi_star = math.sqrt(phi ** 2 + sigma_new ** 2)
        phi_new = 1 / math.sqrt(1 / phi_star ** 2 + 1 / v)
        mu_new = mu + phi_new ** 2 * g * (score - E)

        state["g_mu"], state["g_phi"], state["g_sigma"] = mu_new, phi_new, sigma_new
        state["g_last"] = date


def _parse_ctrl(val):
    """Parses control-time strings like "4:33" into seconds."""
    try:
        minutes, seconds = val.split(":")
        return int(minutes) * 60 + int(seconds)
    except (ValueError, AttributeError):
        return 0


def _build_fighter(name, physical, state, pool_means):
    p = physical.get(name)
    if p is None:
        return None

    minutes = state["rounds"] * 5
    if minutes > 0:
        stats = {
            "slpm":    state["str_landed"] / minutes,
            "str_acc": state["str_landed"] / state["str_attempted"] if state["str_attempted"] else 0.0,
            "sapm":    state["str_absorbed"] / minutes,
            "str_def": 1 - (state["str_absorbed"] / state["opp_str_attempted"]) if state["opp_str_attempted"] else 0.0,
            "td_avg":  state["td_landed"] / minutes * 15,
            "td_acc":  state["td_landed"] / state["td_attempted"] if state["td_attempted"] else 0.0,
            "td_def":  1 - (state["opp_td_landed"] / state["opp_td_attempted"]) if state["opp_td_attempted"] else 0.0,
            "sub_avg": state["sub_att"] / minutes * 15,
        }
    else:
        stats = pool_means  # no fight history yet — treat as an average fighter

    return Fighter(
        first=p["first"], last=p["last"], nickname=p["nickname"],
        height=p["height"], weight=p["weight"], reach=p["reach"],
        stance=p["stance"], dob=p["dob"],
        wins=state["wins"], losses=state["losses"], draws=state["draws"],
        **stats,
    )


def _pre_fight_rates(state):
    """(slpm, sapm) from career totals so far, or None with no cage time yet."""
    minutes = state["rounds"] * 5
    if minutes == 0:
        return None
    return state["str_landed"] / minutes, state["str_absorbed"] / minutes


def _apply_fight_to_state(state_a, state_b, fight, league=None):
    method = fight.get("method", "")
    is_ko = "KO/TKO" in method
    is_finish = is_ko or "Submission" in method

    # Time / context bookkeeping (both fighters, regardless of result).
    # Opponents' Elo is captured BEFORE this fight's Elo update; same for the
    # opponents' striking rates (for opponent-adjusted stats) and own Elo
    # history (for the Elo-trend feature).
    elo_a, elo_b = state_a["elo"], state_b["elo"]
    state_a["opp_elos"].append(elo_b)
    state_b["opp_elos"].append(elo_a)
    state_a["elo_log"].append(elo_a)
    state_b["elo_log"].append(elo_b)
    opp_rates = {id(state_a): _pre_fight_rates(state_b),
                 id(state_b): _pre_fight_rates(state_a)}

    date = fight.get("date")
    weightclass = fight.get("weightclass") or None
    for state in (state_a, state_b):
        if date is not None:
            if state["first_date"] is None:
                state["first_date"] = date
            state["last_date"] = date
        if fight.get("five_round"):
            state["five_round_fights"] += 1
        if weightclass and "Catch" not in weightclass and "Open" not in weightclass:
            state["changed_class"] = state["last_class"] is not None and weightclass != state["last_class"]
            state["last_class"] = weightclass

    if fight["a_result"] == "W":
        winner, loser = state_a, state_b
    elif fight["a_result"] == "L":
        winner, loser = state_b, state_a
    else:
        winner = loser = None

    # Head-to-head history by opponent name, for common-opponent comparisons.
    score_a = 1.0 if fight["a_result"] == "W" else 0.0 if fight["a_result"] == "L" else 0.5
    state_a["opp_results"].setdefault(fight["fighter_b"], []).append(score_a)
    state_b["opp_results"].setdefault(fight["fighter_a"], []).append(1 - score_a)

    if winner is not None:
        winner["wins"] += 1
        loser["losses"] += 1
        winner["streak"] += 1
        loser["streak"] = 0
        winner["recent"].append(1)
        loser["recent"].append(0)
        _update_elo(winner, loser)
        _glicko_update(winner, loser, fight.get("date"))
        if is_finish:
            winner["finish_wins"] += 1
            end_round = fight.get("end_round", 0)
            if end_round == 1:
                winner["r1_finishes"] += 1
            elif end_round >= 3:
                winner["late_finishes"] += 1
        if is_ko:
            loser["ko_losses"] += 1
    else:
        state_a["draws"] += 1
        state_b["draws"] += 1
        state_a["streak"] = 0
        state_b["streak"] = 0
        state_a["recent"].append(0)
        state_b["recent"].append(0)

    rows_by_round = defaultdict(list)
    for row in fight["rounds"]:
        rows_by_round[row["ROUND"]].append(row)

    # Accumulate this fight's totals separately per fighter, then fold them
    # into career totals AND append them to the fighter's fight_log so
    # recent-window rates (e.g. last 3 fights) can be computed later.
    fight_totals = {id(state_a): defaultdict(int), id(state_b): defaultdict(int)}

    for rows in rows_by_round.values():
        if len(rows) != 2:
            continue
        for i, row in enumerate(rows):
            name = row["FIGHTER"].strip().lower()
            state = state_a if name == fight["fighter_a"] else state_b if name == fight["fighter_b"] else None
            if state is None:
                continue
            opponent_row = rows[1 - i]

            landed, attempted = _parse_of(row["SIG.STR."])
            opp_landed, opp_attempted = _parse_of(opponent_row["SIG.STR."])
            td_landed, td_attempted = _parse_of(row["TD"])
            opp_td_landed, opp_td_attempted = _parse_of(opponent_row["TD"])

            ft = fight_totals[id(state)]
            ft["str_landed"] += landed
            ft["str_attempted"] += attempted
            ft["str_absorbed"] += opp_landed
            ft["opp_str_attempted"] += opp_attempted
            ft["td_landed"] += td_landed
            ft["td_attempted"] += td_attempted
            ft["opp_td_landed"] += opp_td_landed
            ft["opp_td_attempted"] += opp_td_attempted
            ft["sub_att"] += _parse_int(row["SUB.ATT"])
            ft["kd"] += _parse_int(row["KD"])
            ft["kd_against"] += _parse_int(opponent_row["KD"])
            ft["ctrl_seconds"] += _parse_ctrl(row["CTRL"])
            ft["opp_ctrl_seconds"] += _parse_ctrl(opponent_row["CTRL"])
            ft["rounds"] += 1

            # per-round output for the cardio signal
            round_num = _parse_int(row["ROUND"].replace("Round", ""))
            if round_num == 1:
                ft["r1_landed"] += landed
                ft["r1_rounds"] += 1
            elif round_num >= 3:
                ft["r3p_landed"] += landed
                ft["r3p_rounds"] += 1

            # position and target mix
            ft["distance_att"] += _parse_of(row["DISTANCE"])[1]
            ft["clinch_att"] += _parse_of(row["CLINCH"])[1]
            ft["ground_att"] += _parse_of(row["GROUND"])[1]
            ft["head_landed"] += _parse_of(row["HEAD"])[0]
            ft["body_landed"] += _parse_of(row["BODY"])[0]
            ft["leg_landed"] += _parse_of(row["LEG"])[0]

    for state in (state_a, state_b):
        ft = fight_totals[id(state)]
        if ft["rounds"] == 0:
            continue
        for key, value in ft.items():
            state[key] += value
            if league is not None:
                league[key] += value
        entry = dict(ft)
        entry["date"] = fight.get("date")  # for time-decayed rates
        # Opponent's pre-fight career striking rates: what a typical fighter
        # lands on / absorbs from them, the baseline for opponent-adjusted
        # output and defense. None when the opponent was a debutant.
        entry["opp_pre"] = opp_rates[id(state)]
        state["fight_log"].append(entry)


# Feature names for the ML dataset, in column order. Every feature is an
# "A minus B" differential (or a symmetric matchup flag), so negating the
# whole vector is exactly the same fight seen from B's corner.
FEATURE_NAMES = [
    "slpm_diff", "str_acc_diff", "sapm_diff", "str_def_diff",
    "td_avg_diff", "td_acc_diff", "td_def_diff", "sub_avg_diff",
    "kd_per_min_diff", "ctrl_share_diff",
    "win_rate_diff", "experience_diff", "streak_diff",
    "elo_diff", "finish_rate_diff", "ko_loss_rate_diff", "recent_form_diff",
    "last3_slpm_diff", "last3_sapm_diff",
    "td_threat", "str_threat",
    "age_diff", "log_layoff_diff", "ring_rust_edge",
    "career_years_diff", "fights_per_year_diff",
    "total_absorbed_diff", "kd_against_rate_diff",
    "opp_quality_diff", "five_round_exp_diff", "class_change_edge",
    "damage_ratio_diff", "r1_finish_rate_diff", "late_finish_rate_diff",
    "standing_share_diff", "cardio_diff", "leg_share_diff", "body_share_diff",
    "reach_diff", "height_diff", "weight_diff",
    "southpaw_edge",
    "adj_output_diff", "adj_defense_diff",
    "slpm_trend_diff", "damage_trend_diff", "elo_trend_diff",
    "age_sq_diff", "age_x_class", "age_x_mileage", "age_x_ko",
    "glicko_diff", "glicko_rd_diff", "rank_score_diff",
    "pre_ufc_exp_diff", "pre_ufc_winrate_diff",
    "red_corner", "h2h_edge", "common_opp_edge", "class_shift_diff",
    # symmetric context (same value from either corner's perspective)
    "common_opp_count",
    "weight_class_ord", "is_women", "is_five_round", "empty_arena",
]

# +1 if a feature is symmetric under swapping the two fighters, -1 if it
# flips sign (all the differentials). Mirroring a row for augmentation or
# order-invariant evaluation means multiplying by this vector, NOT plain
# negation — negating a symmetric feature would corrupt it.
SYMMETRIC_FEATURES = {"common_opp_count", "weight_class_ord", "is_women", "is_five_round", "empty_arena"}
FEATURE_SIGNS = [1.0 if n in SYMMETRIC_FEATURES else -1.0 for n in FEATURE_NAMES]

_WEIGHT_CLASSES = [  # checked in order; "Light Heavyweight" before "Heavyweight"
    "Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
]


def _weightclass_context(wc_string):
    """Maps a bout's WEIGHTCLASS string to (ordinal 1-9, is_women). (0, 0) if unknown."""
    wc = wc_string or ""
    is_women = 1.0 if "Women" in wc else 0.0
    if "Light Heavyweight" in wc:
        return 8.0, is_women
    for i, name in enumerate(_WEIGHT_CLASSES):
        if name in wc:
            return float(i + 1), is_women
    return 0.0, is_women


def _class_shift(bout_wc_ord, last_class):
    """
    How many divisions this bout is above the class the fighter last fought
    at (+1 = moving up one class). 0 when either class is unknown (debuts,
    catchweights, open weight).
    """
    last_ord, _ = _weightclass_context(last_class)
    if not bout_wc_ord or not last_ord:
        return 0.0
    return bout_wc_ord - last_ord


def _rank_score(rank_str):
    """Official ranking → score in [0, 1]: champ/#1 ≈ 1, unranked = 0."""
    try:
        rank = float(rank_str)
    except (ValueError, TypeError):
        return 0.0
    return max((17 - rank) / 16, 0.0) if rank >= 0 else 0.0

MEAN_UFC_AGE = 29.5  # imputed when a fighter's DOB is missing from the data
DECAY_HALF_LIFE_YEARS = 2.0  # a fight 2 years ago counts half as much as one today


def _apply_time_decay(rates, state, as_of, pool_means):
    """
    Recomputes the per-minute/accuracy rates with exponential time decay, so
    a fighter's recent performances dominate their profile and decade-old
    rounds barely register. Overwrites the career-average values in `rates`.
    Uses the same Bayesian shrinkage as the career rates (the decayed counts
    are simply smaller samples). Skipped when dates are unavailable.
    """
    if as_of is None:
        return

    sums = defaultdict(float)
    for entry in state["fight_log"]:
        if entry.get("date") is None:
            return  # mixed/missing dates — stick with career averages
        age_years = max((as_of - entry["date"]).days, 0) / 365.25
        w = 0.5 ** (age_years / DECAY_HALF_LIFE_YEARS)
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                sums[key] += w * value

    minutes = sums["rounds"] * 5
    if minutes <= 0:
        return

    rates.update(_shrunk_rates(sums, minutes, pool_means))
    rates["kd_per_min"] = sums["kd"] / minutes
    rates["ctrl_share"] = sums["ctrl_seconds"] / (minutes * 60)
    rates["kd_against_rate"] = sums["kd_against"] / minutes
    rates["damage_ratio"] = _math_log_ratio(sums["str_landed"], sums["str_absorbed"])


def _math_log_ratio(landed, absorbed):
    import math
    return math.log((landed + 1) / (absorbed + 1))


def _add_time_features(rates, state, phys, as_of, pool_means):
    """
    Augments a fighter's _state_features() dict with features that depend on
    the calendar: age, layoff since last fight, career length, activity.
    `as_of` is the fight date during training/backtesting, or "today" for a
    live prediction. Falls back to neutral values when dates are unavailable.
    """
    import math

    dob = _parse_date(phys.get("dob", ""))
    if dob is not None and as_of is not None:
        rates["age"] = (as_of - dob).days / 365.25
    else:
        rates["age"] = MEAN_UFC_AGE

    if as_of is not None and state["last_date"] is not None:
        layoff_days = max((as_of - state["last_date"]).days, 0)
    else:
        layoff_days = 180  # ~median gap between UFC fights
    rates["log_layoff"] = math.log1p(layoff_days)
    rates["ring_rust"] = 1.0 if layoff_days > 365 else 0.0

    if as_of is not None and state["first_date"] is not None:
        career_years = max((as_of - state["first_date"]).days / 365.25, 0.0)
    else:
        career_years = 0.0
    rates["career_years"] = career_years
    total = state["wins"] + state["losses"] + state["draws"]
    rates["fights_per_year"] = total / career_years if career_years > 0.5 else total

    # Accumulated career damage — total strikes eaten and knockdowns suffered.
    minutes = state["rounds"] * 5
    rates["total_absorbed"] = state["str_absorbed"] / 1000  # scaled to keep magnitudes tame
    rates["kd_against_rate"] = state["kd_against"] / minutes if minutes else 0.0

    opp_elos = state["opp_elos"][-3:]
    rates["opp_quality"] = sum(opp_elos) / len(opp_elos) if opp_elos else 1500.0
    rates["five_round_exp"] = state["five_round_fights"]
    rates["class_change"] = 1.0 if state["changed_class"] else 0.0

    # Glicko-2, on the public scale, with deviation widened to "now" so a
    # long-inactive fighter's rating reads as uncertain, not just stale.
    rates["glicko"] = 1500 + 173.7178 * state["g_mu"]
    phi = state["g_phi"]
    if state["g_last"] is not None and as_of is not None:
        periods = max((as_of - state["g_last"]).days, 0) / 91.3
        phi = min((phi ** 2 + state["g_sigma"] ** 2 * periods) ** 0.5, GLICKO_MAX_PHI)
    rates["glicko_rd"] = 173.7178 * phi

    _apply_time_decay(rates, state, as_of, pool_means)
    return rates


# Bayesian shrinkage priors: every rate/accuracy is pulled toward the pool
# mean with a pseudo-count, so a 2-fight fighter's "70% takedown accuracy"
# reads as slightly-above-average rather than elite, while a 20-fight
# veteran's numbers stand mostly on their own.
#   - accuracies (beta-binomial): prior worth SHRINK_*_ATT attempts
#   - per-minute rates (poisson-gamma): prior worth SHRINK_MINUTES minutes
SHRINK_STR_ATT = 150   # sig. strikes attempted (~1.5 fights of volume)
SHRINK_TD_ATT = 8      # takedown attempts are far rarer than strikes
SHRINK_MINUTES = 30    # ~2 fights' worth of cage time


def _shrunk_ratio(num, den, prior_mean, prior_strength):
    """(num + prior) / (den + prior): shrinks num/den toward prior_mean."""
    return (num + prior_mean * prior_strength) / (den + prior_strength)


def _shrunk_rates(counts, minutes, pool_means):
    """
    The eight core stats from raw counts, each shrunk toward the pool mean.
    `counts` needs the same keys as the running state totals. td_avg/sub_avg
    priors are converted from per-15-min pool means to per-minute space.
    """
    return {
        "slpm":    _shrunk_ratio(counts["str_landed"], minutes, pool_means["slpm"], SHRINK_MINUTES),
        "str_acc": _shrunk_ratio(counts["str_landed"], counts["str_attempted"], pool_means["str_acc"], SHRINK_STR_ATT),
        "sapm":    _shrunk_ratio(counts["str_absorbed"], minutes, pool_means["sapm"], SHRINK_MINUTES),
        "str_def": 1 - _shrunk_ratio(counts["str_absorbed"], counts["opp_str_attempted"], 1 - pool_means["str_def"], SHRINK_STR_ATT),
        "td_avg":  _shrunk_ratio(counts["td_landed"], minutes, pool_means["td_avg"] / 15, SHRINK_MINUTES) * 15,
        "td_acc":  _shrunk_ratio(counts["td_landed"], counts["td_attempted"], pool_means["td_acc"], SHRINK_TD_ATT),
        "td_def":  1 - _shrunk_ratio(counts["opp_td_landed"], counts["opp_td_attempted"], 1 - pool_means["td_def"], SHRINK_TD_ATT),
        "sub_avg": _shrunk_ratio(counts["sub_att"], minutes, pool_means["sub_avg"] / 15, SHRINK_MINUTES) * 15,
    }


def _state_features(state, pool_means):
    """Converts a fighter's running state into per-minute rates and record numbers."""
    minutes = state["rounds"] * 5
    total_fights = state["wins"] + state["losses"] + state["draws"]

    if minutes > 0:
        rates = _shrunk_rates(state, minutes, pool_means)
        rates["kd_per_min"] = state["kd"] / minutes
        rates["ctrl_share"] = state["ctrl_seconds"] / (minutes * 60)
    else:
        rates = dict(pool_means, kd_per_min=0.0, ctrl_share=0.0)

    rates["win_rate"] = state["wins"] / total_fights if total_fights else 0.5
    rates["experience"] = total_fights
    rates["streak"] = state["streak"]
    rates["elo"] = state["elo"]
    rates["finish_rate"] = state["finish_wins"] / total_fights if total_fights else 0.0
    rates["ko_loss_rate"] = state["ko_losses"] / total_fights if total_fights else 0.0
    import math as _math

    # Style / cardio / finishing profile
    rates["damage_ratio"] = _math.log((state["str_landed"] + 1) / (state["str_absorbed"] + 1))
    rates["r1_finish_rate"] = state["r1_finishes"] / total_fights if total_fights else 0.0
    rates["late_finish_rate"] = state["late_finishes"] / total_fights if total_fights else 0.0

    position_att = state["distance_att"] + state["clinch_att"] + state["ground_att"]
    rates["standing_share"] = state["distance_att"] / position_att if position_att else 0.75

    if state["r1_rounds"] and state["r3p_rounds"] and state["r1_landed"]:
        r1_rate = state["r1_landed"] / state["r1_rounds"]
        r3p_rate = state["r3p_landed"] / state["r3p_rounds"]
        # >1: gets stronger late; <1: fades. Clipped — tiny round samples
        # can produce absurd ratios that would dominate the fit.
        rates["cardio"] = min(max(r3p_rate / r1_rate, 0.25), 3.0)
    else:
        rates["cardio"] = 1.0

    landed = state["str_landed"]
    rates["leg_share"] = state["leg_landed"] / landed if landed else 0.0
    rates["body_share"] = state["body_landed"] / landed if landed else 0.0

    # Opponent-adjusted striking: each fight's output measured against what
    # that opponent typically absorbs (and absorption against what that
    # opponent typically lands), instead of against the whole field. >1 means
    # outperforming expectation. Ratios are clipped — a debut opponent or a
    # 30-second fight can produce absurd values.
    ratios_out, ratios_def = [], []
    for entry in state["fight_log"]:
        opp = entry.get("opp_pre")
        if not opp or not entry["rounds"]:
            continue
        opp_slpm, opp_sapm = opp
        fight_minutes = entry["rounds"] * 5
        ratios_out.append(min((entry["str_landed"] / fight_minutes) / max(opp_sapm, 0.5), 4.0))
        ratios_def.append(min((entry["str_absorbed"] / fight_minutes) / max(opp_slpm, 0.5), 4.0))
    rates["adj_output"] = sum(ratios_out) / len(ratios_out) if ratios_out else 1.0
    rates["adj_defense"] = sum(ratios_def) / len(ratios_def) if ratios_def else 1.0

    # Trajectory: is the fighter improving or declining? Recent-window levels
    # (last3_*) miss direction — a fighter at 4.0 SLpM on the way down from
    # 5.5 is different from one on the way up from 2.5.
    last5 = [e for e in state["fight_log"][-5:] if e["rounds"]]
    rates["slpm_trend"] = _slope([e["str_landed"] / (e["rounds"] * 5) for e in last5])
    rates["damage_trend"] = _slope([_math_log_ratio(e["str_landed"], e["str_absorbed"]) for e in last5])
    elo_log = state["elo_log"]
    rates["elo_trend"] = (state["elo"] - elo_log[-3]) / 3 if len(elo_log) >= 3 else 0.0

    recent = state["recent"][-3:]
    rates["recent_form"] = sum(recent) / len(recent) if recent else 0.5

    # Recent-window striking rates: career averages hide decline/improvement,
    # so also expose per-minute rates over the last 3 fights only.
    last3 = state["fight_log"][-3:]
    last3_minutes = sum(f["rounds"] for f in last3) * 5
    if last3_minutes > 0:
        rates["last3_slpm"] = sum(f["str_landed"] for f in last3) / last3_minutes
        rates["last3_sapm"] = sum(f["str_absorbed"] for f in last3) / last3_minutes
    else:
        rates["last3_slpm"] = rates["slpm"]
        rates["last3_sapm"] = rates["sapm"]
    return rates


def _matchup_history_features(state_a, state_b, name_a, name_b):
    """
    (h2h_edge, common_opp_edge, common_opp_count) for a matchup:
      - h2h_edge: A's average result in prior meetings with B, scaled to
        [-1, 1] (positive = A has beaten B before). 0 with no history.
      - common_opp_edge: A's average result minus B's average result against
        the opponents they share. 0 with no shared opponents.
      - common_opp_count: how many opponents they share (symmetric).
    """
    h2h = state_a["opp_results"].get(name_b)
    h2h_edge = (sum(h2h) / len(h2h) - 0.5) * 2 if h2h else 0.0

    shared = set(state_a["opp_results"]) & set(state_b["opp_results"])
    if shared:
        mean_a = sum(sum(r) / len(r) for o in shared for r in [state_a["opp_results"][o]]) / len(shared)
        mean_b = sum(sum(r) / len(r) for o in shared for r in [state_b["opp_results"][o]]) / len(shared)
        edge = mean_a - mean_b
    else:
        edge = 0.0
    return h2h_edge, edge, float(len(shared))


def _slope(values):
    """Least-squares slope of values over their index (per-fight trend)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2
    mean_y = sum(values) / n
    denom = sum((i - mean_x) ** 2 for i in range(n))
    return sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / denom


def _stance_edge(stance_a, stance_b):
    a, b = (stance_a or "").lower(), (stance_b or "").lower()
    if a == "southpaw" and b == "orthodox":
        return 1.0
    if b == "southpaw" and a == "orthodox":
        return -1.0
    return 0.0


DEFAULT_CONTEXT = {
    "rank_a": 0.0, "rank_b": 0.0,       # official-ranking scores (0 = unranked)
    "red_corner": 0.0,                  # +1 if A is the red corner, -1 blue, 0 unknown
    "h2h_edge": 0.0,                    # A's prior record vs B, in [-1, 1]
    "common_opp_edge": 0.0,             # A minus B, vs shared opponents
    "class_shift_a": 0.0,               # THIS bout's class minus the class A last fought at
    "class_shift_b": 0.0,               # (+1 = moving up one division for this fight)
    "common_opp_count": 0.0,            # how many opponents they share (symmetric)
    "weight_class_ord": 0.0,            # 1 (strawweight) … 9 (heavyweight); 0 unknown
    "is_women": 0.0,
    "is_five_round": 0.0,
    "empty_arena": 0.0,
}


def build_feature_row(fa, fb, phys_a, phys_b, context=None):
    """
    Builds the FEATURE_NAMES differential vector for a matchup, given each
    fighter's _state_features() dict and physical profile. Shared by training
    (generate_dataset) and live prediction so the two can never drift apart.
    The betting line is deliberately NOT a feature — the model is pure stats,
    and the market is blended in afterwards (see model.predict_proba).
    context carries fight-level info (see DEFAULT_CONTEXT); note rank_a/rank_b
    are per-fighter and red_corner/h2h_edge/common_opp_edge are signed from
    A's perspective, so callers swapping a/b must swap/negate those too.
    """
    ctx = {**DEFAULT_CONTEXT, **(context or {})}
    return [
        fa["slpm"] - fb["slpm"],
        fa["str_acc"] - fb["str_acc"],
        fa["sapm"] - fb["sapm"],
        fa["str_def"] - fb["str_def"],
        fa["td_avg"] - fb["td_avg"],
        fa["td_acc"] - fb["td_acc"],
        fa["td_def"] - fb["td_def"],
        fa["sub_avg"] - fb["sub_avg"],
        fa["kd_per_min"] - fb["kd_per_min"],
        fa["ctrl_share"] - fb["ctrl_share"],
        fa["win_rate"] - fb["win_rate"],
        fa["experience"] - fb["experience"],
        fa["streak"] - fb["streak"],
        fa["elo"] - fb["elo"],
        fa["finish_rate"] - fb["finish_rate"],
        fa["ko_loss_rate"] - fb["ko_loss_rate"],
        fa["recent_form"] - fb["recent_form"],
        fa["last3_slpm"] - fb["last3_slpm"],
        fa["last3_sapm"] - fb["last3_sapm"],
        # matchup interactions: my offense vs YOUR defense, not vs the field
        fa["td_avg"] * (1 - fb["td_def"]) - fb["td_avg"] * (1 - fa["td_def"]),
        fa["slpm"] * (1 - fb["str_def"]) - fb["slpm"] * (1 - fa["str_def"]),
        fa["age"] - fb["age"],
        fa["log_layoff"] - fb["log_layoff"],
        fa["ring_rust"] - fb["ring_rust"],
        fa["career_years"] - fb["career_years"],
        fa["fights_per_year"] - fb["fights_per_year"],
        fa["total_absorbed"] - fb["total_absorbed"],
        fa["kd_against_rate"] - fb["kd_against_rate"],
        fa["opp_quality"] - fb["opp_quality"],
        fa["five_round_exp"] - fb["five_round_exp"],
        fa["class_change"] - fb["class_change"],
        fa["damage_ratio"] - fb["damage_ratio"],
        fa["r1_finish_rate"] - fb["r1_finish_rate"],
        fa["late_finish_rate"] - fb["late_finish_rate"],
        fa["standing_share"] - fb["standing_share"],
        fa["cardio"] - fb["cardio"],
        fa["leg_share"] - fb["leg_share"],
        fa["body_share"] - fb["body_share"],
        phys_a["reach"] - phys_b["reach"],
        phys_a["height"] - phys_b["height"],
        phys_a["weight"] - phys_b["weight"],
        _stance_edge(phys_a["stance"], phys_b["stance"]),
        fa["adj_output"] - fb["adj_output"],
        fa["adj_defense"] - fb["adj_defense"],
        fa["slpm_trend"] - fb["slpm_trend"],
        fa["damage_trend"] - fb["damage_trend"],
        fa["elo_trend"] - fb["elo_trend"],
        # Age effects are nonlinear (the decline past ~35 is a cliff, not a
        # line) and interact with weight class and accumulated damage.
        fa["age"] ** 2 - fb["age"] ** 2,
        (fa["age"] - fb["age"]) * ctx["weight_class_ord"],
        fa["age"] * fa["total_absorbed"] - fb["age"] * fb["total_absorbed"],
        fa["age"] * fa["ko_loss_rate"] - fb["age"] * fb["ko_loss_rate"],
        fa["glicko"] - fb["glicko"],
        fa["glicko_rd"] - fb["glicko_rd"],
        ctx["rank_a"] - ctx["rank_b"],
        _pre_ufc_features(phys_a)[0] - _pre_ufc_features(phys_b)[0],
        _pre_ufc_features(phys_a)[1] - _pre_ufc_features(phys_b)[1],
        ctx["red_corner"],
        ctx["h2h_edge"],
        ctx["common_opp_edge"],
        # Moving up a class for THIS bout (stats earned against smaller men)
        # vs the opponent doing so — unlike class_change_edge, which only
        # knows about a switch made one fight ago.
        ctx["class_shift_a"] - ctx["class_shift_b"],
        ctx["common_opp_count"],
        ctx["weight_class_ord"],
        ctx["is_women"],
        ctx["is_five_round"],
        ctx["empty_arena"],
    ]


def _compute_pool_means(data_dir):
    from src.loader import load_fighters
    norm_stats = _compute_norm_stats(load_fighters(data_dir))
    return {stat: norm_stats[stat][0] for stat in
            ("slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg")}


LEAGUE_MIN_MINUTES = 5000  # ~2 years of early UFC before to-date means are trusted


def _league_pool_means(league, fallback):
    """
    League-wide average stats from the fights replayed SO FAR, used as the
    debut-fill values and shrinkage priors. Using to-date means instead of
    means over the whole (future-inclusive) dataset both removes a subtle
    leak and tracks era drift — league-wide striking volume has risen for
    two decades, so 2005 debuts shouldn't be filled with 2026 averages.
    Falls back to the static pool means until enough history accumulates.
    League totals sum both corners, so landed == absorbed by construction.
    """
    minutes = league["rounds"] * 5
    if minutes < LEAGUE_MIN_MINUTES:
        return fallback
    return {
        "slpm":    league["str_landed"] / minutes,
        "str_acc": league["str_landed"] / league["str_attempted"],
        "sapm":    league["str_absorbed"] / minutes,
        "str_def": 1 - league["str_absorbed"] / league["opp_str_attempted"],
        "td_avg":  league["td_landed"] / minutes * 15,
        "td_acc":  league["td_landed"] / league["td_attempted"] if league["td_attempted"] else fallback["td_acc"],
        "td_def":  1 - league["opp_td_landed"] / league["opp_td_attempted"] if league["opp_td_attempted"] else fallback["td_def"],
        "sub_avg": league["sub_att"] / minutes * 15,
    }


def build_current_states(data_dir):
    """
    Replays all of UFC history and returns (states, physical, pool_means):
    each fighter's up-to-date running state after their entire career so far.
    This is what live prediction should use — the same state representation
    the model was trained on, just evaluated at "now".
    """
    physical = load_physical_profiles(data_dir)
    _attach_pre_ufc_records(physical, data_dir)
    fallback_means = _compute_pool_means(data_dir)
    states = defaultdict(_new_state)
    league = defaultdict(int)

    for fight in _load_fights_chronological(data_dir):
        _apply_fight_to_state(states[fight["fighter_a"]], states[fight["fighter_b"]], fight, league)

    return states, physical, _league_pool_means(league, fallback_means)


def generate_dataset(data_dir, with_meta=False):
    """
    Replays every fight chronologically (same leakage-free discipline as
    run_backtest) and returns (X, y) where each row of X is the FEATURE_NAMES
    differential vector for one fight and y is 1 if fighter A won.

    With with_meta=True, also returns a third list (aligned with the rows) of
    {fighter_a, fighter_b, date, odds_a, odds_b} — odds are American odds, or
    None when the fight had no betting line.

    Fights where either fighter has no prior UFC history are excluded —
    there is nothing real to learn from a debut.

    Rows come out in chronological order so callers can split train/test by
    time (train on the past, test on the future) instead of randomly.
    """
    physical = load_physical_profiles(data_dir)
    _attach_pre_ufc_records(physical, data_dir)
    fallback_means = _compute_pool_means(data_dir)
    fights = _load_fights_chronological(data_dir)
    odds = _load_odds(data_dir)

    states = defaultdict(_new_state)
    league = defaultdict(int)
    X, y, meta = [], [], []

    for fight in fights:
        pool_means = _league_pool_means(league, fallback_means)
        name_a, name_b = fight["fighter_a"], fight["fighter_b"]
        state_a, state_b = states[name_a], states[name_b]
        phys_a, phys_b = physical.get(name_a), physical.get(name_b)

        total_a = state_a["wins"] + state_a["losses"] + state_a["draws"]
        total_b = state_b["wins"] + state_b["losses"] + state_b["draws"]

        # A fighter is predictable if they have UFC history OR a known
        # pre-UFC record — this brings previously-excluded debut fights in.
        known_a = total_a > 0 or (phys_a and _pre_ufc_features(phys_a)[0] > 0)
        known_b = total_b > 0 or (phys_b and _pre_ufc_features(phys_b)[0] > 0)

        if phys_a and phys_b and known_a and known_b and fight["a_result"] != "D":
            fa = _add_time_features(_state_features(state_a, pool_means), state_a, phys_a, fight["date"], pool_means)
            fb = _add_time_features(_state_features(state_b, pool_means), state_b, phys_b, fight["date"], pool_means)
            line = _find_odds(odds, fight["date"], name_a, name_b)
            wc_ord, is_women = _weightclass_context(fight.get("weightclass"))
            h2h_edge, co_edge, co_count = _matchup_history_features(state_a, state_b, name_a, name_b)
            context = {
                "rank_a": line["rank"][name_a] if line else 0.0,
                "rank_b": line["rank"][name_b] if line else 0.0,
                "red_corner": (1.0 if line["red"] == name_a else -1.0) if line else 0.0,
                "h2h_edge": h2h_edge,
                "common_opp_edge": co_edge,
                "class_shift_a": _class_shift(wc_ord, state_a["last_class"]),
                "class_shift_b": _class_shift(wc_ord, state_b["last_class"]),
                "common_opp_count": co_count,
                "weight_class_ord": wc_ord,
                "is_women": is_women,
                "is_five_round": 1.0 if fight.get("five_round") else 0.0,
                "empty_arena": line["empty_arena"] if line else 0.0,
            }
            X.append(build_feature_row(fa, fb, phys_a, phys_b, context))
            y.append(1 if fight["a_result"] == "W" else 0)
            if with_meta:
                meta.append({
                    "fighter_a": name_a, "fighter_b": name_b, "date": fight["date"],
                    "odds_a": line["american"][name_a] if line else None,
                    "odds_b": line["american"][name_b] if line else None,
                    "debut": total_a == 0 or total_b == 0,
                    "method": fight["method"],
                })

        _apply_fight_to_state(state_a, state_b, fight, league)

    if with_meta:
        return X, y, meta
    return X, y


def run_backtest(data_dir):
    """
    Replays every fight in chronological order and checks whether the
    predictor (using only pre-fight data) would have picked the real winner.

    Returns a dict with accuracy stats.
    """
    physical = load_physical_profiles(data_dir)
    fights = _load_fights_chronological(data_dir)

    # Normalization stats (mean/std per feature) are computed once from the
    # final, full career dataset via the real loader. This is an approximation:
    # it uses "today's" league-wide averages even for fights decided decades
    # ago. Recomputing them at every point in time would be more correct but
    # is a lot more expensive; this is a reasonable first cut.
    from src.loader import load_fighters
    norm_stats = _compute_norm_stats(load_fighters(data_dir))
    pool_means = {stat: norm_stats[stat][0] for stat in
                  ("slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg")}

    states = defaultdict(_new_state)

    total = 0
    correct = 0
    baseline_correct = 0  # naive baseline: pick whoever has more career wins so far
    skipped_cold_start = 0

    for fight in fights:
        name_a, name_b = fight["fighter_a"], fight["fighter_b"]
        state_a, state_b = states[name_a], states[name_b]

        fighter_a = _build_fighter(name_a, physical, state_a, pool_means)
        fighter_b = _build_fighter(name_b, physical, state_b, pool_means)

        has_history = state_a["wins"] + state_a["losses"] + state_a["draws"] > 0 and \
                      state_b["wins"] + state_b["losses"] + state_b["draws"] > 0

        if fighter_a is None or fighter_b is None or not has_history:
            skipped_cold_start += 1
        else:
            score_a = compute_score(fighter_a, norm_stats)
            score_b = compute_score(fighter_b, norm_stats)

            f1_stance = (fighter_a.stance or "").lower()
            f2_stance = (fighter_b.stance or "").lower()
            if f1_stance == "southpaw" and f2_stance == "orthodox":
                score_a += STANCE_BONUS
            elif f2_stance == "southpaw" and f1_stance == "orthodox":
                score_b += STANCE_BONUS

            predicted_a_wins = score_a >= score_b
            actual_a_wins = fight["a_result"] == "W"

            if fight["a_result"] != "D":  # draws aren't a meaningful "did we pick the winner" case
                total += 1
                if predicted_a_wins == actual_a_wins:
                    correct += 1

                baseline_a_wins = state_a["wins"] >= state_b["wins"]
                if baseline_a_wins == actual_a_wins:
                    baseline_correct += 1

        _apply_fight_to_state(state_a, state_b, fight)

    return {
        "total_fights": len(fights),
        "evaluated": total,
        "skipped_cold_start": skipped_cold_start,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0.0,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": round(baseline_correct / total * 100, 2) if total else 0.0,
    }


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results = run_backtest(data_dir)
    print(f"Total fights in dataset:     {results['total_fights']}")
    print(f"Skipped (cold start):        {results['skipped_cold_start']}")
    print(f"Evaluated:                   {results['evaluated']}")
    print(f"Model accuracy:              {results['accuracy']}% ({results['correct']}/{results['evaluated']})")
    print(f"Baseline (more wins so far):  {results['baseline_accuracy']}% ({results['baseline_correct']}/{results['evaluated']})")
