from src.fighter import Fighter
from collections import defaultdict
import csv


def _parse_of(val):
    """Parses UFC-stats "X of Y" strings (e.g. "15 of 30") into (landed, attempted) ints."""
    try:
        landed, attempted = val.split(" of ")
        return int(landed), int(attempted)
    except (ValueError, AttributeError):
        return 0, 0


def _parse_int(val):
    # Some numeric columns hold float strings (e.g. "1.0"), which int() rejects.
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _parse_height(val):
    """Parses e.g. 6' 4\" into total inches. Returns None if missing/unknown."""
    val = (val or "").strip()
    if not val or val == "--":
        return None
    try:
        feet, inches = val.replace('"', "").split("'")
        return int(feet.strip()) * 12 + int(inches.strip())
    except ValueError:
        return None


def _parse_weight(val):
    """Parses e.g. "185 lbs." into 185.0. Returns None if missing/unknown."""
    val = (val or "").strip()
    if not val or val == "--":
        return None
    try:
        return float(val.replace("lbs.", "").replace("lbs", "").strip())
    except ValueError:
        return None


def _parse_reach(val):
    """Parses e.g. 74.0" into 74.0. Returns None if missing/unknown."""
    val = (val or "").strip()
    if not val or val == "--":
        return None
    try:
        return float(val.replace('"', ""))
    except ValueError:
        return None


def _fill_missing_with_mean(raw_by_name, field):
    """Replaces None values for `field` with the mean of all known values for that field."""
    known = [r[field] for r in raw_by_name.values() if r[field] is not None]
    mean = sum(known) / len(known) if known else 0.0
    for r in raw_by_name.values():
        if r[field] is None:
            r[field] = mean


def load_physical_profiles(data_dir):
    """
    Reads ufc_fighter_details.csv + ufc_fighter_tott.csv and returns
    { full_name: {first, last, nickname, height, weight, reach, stance, dob} }.

    These attributes are static for a fighter's whole career (unlike the
    record/stats, which change fight-to-fight), so this is split out to be
    reusable by anything that needs to replay fight history chronologically.
    """
    # --- Load fighter details (name + URL) ---
    details = {}  # { url: {FIRST, LAST, NICKNAME} }

    with open(f"{data_dir}/ufc_fighter_details.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            details[row["URL"]] = {
                "first": row["FIRST"],
                "last": row["LAST"],
                "nickname": row["NICKNAME"]
            }

    # --- Load fighter physical stats and join with details ---
    fighters_raw = {}  # { full_name: {all combined fields} }

    with open(f"{data_dir}/ufc_fighter_tott.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["URL"]

            # Only process if we have matching details for this URL
            if url not in details:
                continue

            d = details[url]
            full_name = f"{d['first']} {d['last']}".strip().lower()

            fighters_raw[full_name] = {
                "first":    d["first"],
                "last":     d["last"],
                "nickname": d["nickname"],
                "height":   _parse_height(row["HEIGHT"]),
                "weight":   _parse_weight(row["WEIGHT"]),
                "reach":    _parse_reach(row["REACH"]),
                "stance":   row["STANCE"],
                "dob":      row["DOB"]
            }

    # Physical attributes are sometimes missing ("--") in the source data.
    # Impute with the pool mean so a missing attribute reads as "average" rather
    # than crashing or dragging the fighter's score down as if it were 0.
    for field in ("height", "weight", "reach"):
        _fill_missing_with_mean(fighters_raw, field)

    return fighters_raw


def load_fighters(data_dir):
    """
    Reads all four CSVs from data_dir, joins the data,
    and returns a dict of {full_name: Fighter} objects.
    """

    fighters_raw = load_physical_profiles(data_dir)

    # --- Step 3: Count wins/losses/draws from fight results ---
    records = {}  # { full_name: {"wins": 0, "losses": 0, "draws": 0} }

    with open(f"{data_dir}/ufc_fight_results.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bout    = row["BOUT"]
            outcome = row["OUTCOME"].strip()

            # Split "Fighter A vs. Fighter B" into two names
            if " vs. " not in bout:
                continue
            fighter_a, fighter_b = bout.split(" vs. ")
            fighter_a = fighter_a.strip().lower()
            fighter_b = fighter_b.strip().lower()

            # Initialise record dicts if we haven't seen these fighters yet
            for name in [fighter_a, fighter_b]:
                if name not in records:
                    records[name] = {"wins": 0, "losses": 0, "draws": 0}

            # Assign result based on outcome
            if outcome == "D":
                records[fighter_a]["draws"] += 1
                records[fighter_b]["draws"] += 1
            elif "/" in outcome:
                a_result, b_result = outcome.split("/")
                if a_result == "W":
                    records[fighter_a]["wins"]   += 1
                    records[fighter_b]["losses"] += 1
                elif a_result == "L":
                    records[fighter_a]["losses"] += 1
                    records[fighter_b]["wins"]   += 1

    # --- Step 4: Aggregate striking/grappling stats per fighter ---
    # Each row is one fighter's performance in one round of one bout. To get
    # rates that need the *opponent's* numbers too (strikes absorbed, defence),
    # we first group rows by (BOUT, ROUND) so each fighter's row can be paired
    # with their opponent's row for that same round.
    rounds_by_key = defaultdict(list)

    with open(f"{data_dir}/ufc_fight_stats.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds_by_key[(row["BOUT"], row["ROUND"])].append(row)

    stats_totals = defaultdict(lambda: {
        "str_landed": 0, "str_attempted": 0,
        "str_absorbed": 0, "opp_str_attempted": 0,
        "td_landed": 0, "td_attempted": 0,
        "opp_td_landed": 0, "opp_td_attempted": 0,
        "sub_att": 0, "rounds": 0,
    })

    for rows in rounds_by_key.values():
        # A round should have exactly one row per side. Skip anything malformed.
        if len(rows) != 2:
            continue

        for i, row in enumerate(rows):
            name = row["FIGHTER"].strip().lower()
            if not name:
                continue
            opponent_row = rows[1 - i]

            landed, attempted = _parse_of(row["SIG.STR."])
            opp_landed, opp_attempted = _parse_of(opponent_row["SIG.STR."])
            td_landed, td_attempted = _parse_of(row["TD"])
            opp_td_landed, opp_td_attempted = _parse_of(opponent_row["TD"])

            s = stats_totals[name]
            s["str_landed"]       += landed
            s["str_attempted"]    += attempted
            s["str_absorbed"]     += opp_landed
            s["opp_str_attempted"] += opp_attempted
            s["td_landed"]        += td_landed
            s["td_attempted"]     += td_attempted
            s["opp_td_landed"]    += opp_td_landed
            s["opp_td_attempted"] += opp_td_attempted
            s["sub_att"]          += _parse_int(row["SUB.ATT"])
            s["rounds"]           += 1

    # Convert raw totals into rates. Rounds are assumed to be 5 minutes each
    # (the source data doesn't record actual round duration).
    computed_stats = {}
    for full_name, s in stats_totals.items():
        minutes = s["rounds"] * 5
        if minutes == 0:
            continue

        computed_stats[full_name] = {
            "slpm":    round(s["str_landed"] / minutes, 2),
            "str_acc": round(s["str_landed"] / s["str_attempted"], 2) if s["str_attempted"] else 0.0,
            "sapm":    round(s["str_absorbed"] / minutes, 2),
            "str_def": round(1 - (s["str_absorbed"] / s["opp_str_attempted"]), 2) if s["opp_str_attempted"] else 0.0,
            "td_avg":  round(s["td_landed"] / minutes * 15, 2),
            "td_acc":  round(s["td_landed"] / s["td_attempted"], 2) if s["td_attempted"] else 0.0,
            "td_def":  round(1 - (s["opp_td_landed"] / s["opp_td_attempted"]), 2) if s["opp_td_attempted"] else 0.0,
            "sub_avg": round(s["sub_att"] / minutes * 15, 2),
        }

    # Fighters with no usable rows in ufc_fight_stats.csv get the pool average
    # for each stat, so they read as "average" instead of artificially 0.
    stat_fields = ["slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]
    pool_means = {}
    for field in stat_fields:
        values = [c[field] for c in computed_stats.values()]
        pool_means[field] = sum(values) / len(values) if values else 0.0

    # --- Step 5: Build Fighter objects ---
    fighters = {}  # { full_name: Fighter } — the final return value

    for full_name, raw in fighters_raw.items():

        rec = records.get(full_name, {"wins": 0, "losses": 0, "draws": 0})
        stats = computed_stats.get(full_name, pool_means)

        fighters[full_name] = Fighter(
            first    = raw["first"],
            last     = raw["last"],
            nickname = raw["nickname"],
            height   = raw["height"],
            weight   = raw["weight"],
            reach    = raw["reach"],
            stance   = raw["stance"],
            dob      = raw["dob"],
            wins     = rec["wins"],
            losses   = rec["losses"],
            draws    = rec["draws"],
            slpm     = stats["slpm"],
            str_acc  = stats["str_acc"],
            sapm     = stats["sapm"],
            str_def  = stats["str_def"],
            td_avg   = stats["td_avg"],
            td_acc   = stats["td_acc"],
            td_def   = stats["td_def"],
            sub_avg  = stats["sub_avg"],
        )

    return fighters
