# display.py
# handles all terminal output for the fight prediction result.
#takes the result dictionary from predictor.predict() and prints it cleanly.

from src.fighter import Fighter

def _print_divider():
    # Prints a horizontal line to visually separate sections of the output.
    print("=" * 50)

def display_prediction(result: dict) -> None:
    _print_divider()

    fighter1 = result["fighter1"]
    fighter2 = result["fighter2"]

    print(f"{'UFC FIGHT PREDICTION':^50}")
    print(f"{fighter1.first} {fighter1.last:^50} VS  {fighter2.first} {fighter2.last}")

    _print_divider()

    # Pull the winner and confidence score from the result dictionary
    winner = result["winner"]
    confidence = result["confidence"]

    # Print the predicted winner
    print(f"{'PREDICTED WINNER':^50}")
    print(f"{winner.first} {winner.last:^50}")

    # Print the confidence score, formatted to one decimal place
    print(f"{'Confidence: ' + str(round(confidence, 1)) + '%':^50}")

    _print_divider()

    # Print a stats comparison section for both fighters
    print(f"{'FIGHTER STATS':^50}")
    print(f"{'':^20} {fighter1.first + ' ' + fighter1.last:^12} {fighter2.first + ' ' + fighter2.last:^12}")
    print()

    # Record
    f1_record = f"{fighter1.wins}-{fighter1.losses}-{fighter1.draws}"
    f2_record = f"{fighter2.wins}-{fighter2.losses}-{fighter2.draws}"
    print(f"{'Record':<20} {f1_record:^12} {f2_record:^12}")

    # Key striking and grappling stats
    print(f"{'Str. Landed/Min':<20} {fighter1.slpm:^12.2f} {fighter2.slpm:^12.2f}")
    print(f"{'Str. Accuracy':<20} {fighter1.str_acc:^12.2f} {fighter2.str_acc:^12.2f}")
    print(f"{'Str. Defence':<20} {fighter1.str_def:^12.2f} {fighter2.str_def:^12.2f}")
    print(f"{'Takedowns/15min':<20} {fighter1.td_avg:^12.2f} {fighter2.td_avg:^12.2f}")
    print(f"{'TD Accuracy':<20} {fighter1.td_acc:^12.2f} {fighter2.td_acc:^12.2f}")
    print(f"{'TD Defence':<20} {fighter1.td_def:^12.2f} {fighter2.td_def:^12.2f}")
    print(f"{'Sub. Avg/15min':<20} {fighter1.sub_avg:^12.2f} {fighter2.sub_avg:^12.2f}")

    _print_divider()

    # Print the raw scores so the user can see the underlying numbers
    score1 = result["score1"]
    score2 = result["score2"]

    print(f"{'RAW SCORES':^50}")
    print(f"{fighter1.first + ' ' + fighter1.last:^25}{fighter2.first + ' ' + fighter2.last:^25}")
    print(f"{round(score1, 4):^25}{round(score2, 4):^25}")

    _print_divider()





