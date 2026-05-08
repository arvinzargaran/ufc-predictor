from src.fighter import Fighter
import csv

def load_fighters(data_dir):
    """
    Reads all four CSVs from data_dir, joins the data,
    and returns a dict of {full_name: Fighter} objects.
    """

    # --- Step 1: Load fighter details (name + URL) ---
    details = {}  # { url: {FIRST, LAST, NICKNAME} }

    with open(f"{data_dir}/ufc_fighter_details.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            details[row["URL"]] = {
                "first": row["FIRST"],
                "last": row["LAST"],
                "nickname": row["NICKNAME"]
            }

    # --- Step 2: Load fighter physical stats and join with details ---
    fighters_raw = {}  # { full_name: {all combined fields} }

    with open(f"{data_dir}/ufc_fighter_tott.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["URL"]

            # Only process if we have matching details for this URL
            if url not in details:
                continue

            d = details[url]
            full_name = f"{d['first']} {d['last']}".strip()

            fighters_raw[full_name] = {
                "first":    d["first"],
                "last":     d["last"],
                "nickname": d["nickname"],
                "height":   row["HEIGHT"],
                "weight":   row["WEIGHT"],
                "reach":    row["REACH"],
                "stance":   row["STANCE"],
                "dob":      row["DOB"]
            }
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
            fighter_a = fighter_a.strip()
            fighter_b = fighter_b.strip()

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
    # Each fighter appears once per round fought, so we average across all rows

    stats_totals = {}  # { full_name: { stat: total, "count": n } }

    with open(f"{data_dir}/ufc_fight_stats.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["FIGHTER"].strip()
            if not name:
                continue

            if name not in stats_totals:
                stats_totals[name] = {
                    "slpm":    0.0,
                    "str_acc": 0.0,
                    "sapm":    0.0,
                    "str_def": 0.0,
                    "td_avg":  0.0,
                    "td_acc":  0.0,
                    "td_def":  0.0,
                    "sub_avg": 0.0,
                    "count":   0
                }

            def parse_pct(val):
                # Converts "72%" → 0.72, handles missing values
                try:
                    return float(val.replace("%", "")) / 100
                except:
                    return 0.0

            def parse_float(val):
                try:
                    return float(val)
                except:
                    return 0.0

            s = stats_totals[name]
            s["slpm"]    += parse_float(row["SIG.STR."])
            s["str_acc"] += parse_pct(row["SIG.STR. %"])
            s["sapm"]    += parse_float(row["SIG.STR."])
            s["td_avg"]  += parse_float(row["TD"])
            s["td_acc"]  += parse_pct(row["TD %"])
            s["sub_avg"] += parse_float(row["SUB.ATT"])
            s["count"]   += 1

    # --- Step 5: Build Fighter objects ---
    fighters = {}  # { full_name: Fighter } — the final return value

    for full_name, raw in fighters_raw.items():

        # Get win/loss/draw record, defaulting to 0 if fighter not in results
        rec = records.get(full_name, {"wins": 0, "losses": 0, "draws": 0})

        # Get stats, defaulting to empty if fighter not in stats csv
        s = stats_totals.get(full_name)
        if s and s["count"] > 0:
            n = s["count"]
            slpm    = round(s["slpm"]    / n, 2)
            str_acc = round(s["str_acc"] / n, 2)
            sapm    = round(s["sapm"]    / n, 2)
            td_avg  = round(s["td_avg"]  / n, 2)
            td_acc  = round(s["td_acc"]  / n, 2)
            sub_avg = round(s["sub_avg"] / n, 2)
        else:
            slpm = str_acc = sapm = td_avg = td_acc = sub_avg = 0.0

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
            slpm     = slpm,
            str_acc  = str_acc,
            sapm     = sapm,
            td_avg   = td_avg,
            td_acc   = td_acc,
            sub_avg  = sub_avg
        )

    return fighters