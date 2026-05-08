#src/fighter.py
#defines the Fighter class which is a data container representing a single UFC fighter.
#This class stores biographical, physical and statistical data for one fighter.
#Data is assembled externally by loader.py and passed into this class via __init__.

class Fighter:
    """
    Represents a single UFC fighter.

    Stores all relevant data for one fighter assembled from multiple CSV sources:
       - ufc_fighter_details.csv : identity (first, last, nickname)
       - ufc_fighter_tott.csv : physical profile (height, weight, reach, stance, DOB)
       - ufc_fight_results.csv :win/loss/draw record (aggregated in loader.py)
       - ufc_fight_stats.csv : career performance statistics (aggregated in loader.py)

     All data is passed in construction time by loader.py.
    """

    def __init__(
            self,
            first, last, nickname,
            height, weight, reach, stance, dob,
            wins = 0, losses = 0, draws = 0,
            slpm = 0.0, str_acc = 0.0, sapm = 0.0, str_def = 0.0,
            td_avg = 0.0, td_acc = 0.0, td_def = 0.0, sub_avg = 0.0
    ):
        """
        Constructs a Fighter object.

        Required args (identity + physical):
            first, last : str -- fighter's first and last name
            nickname    : str -- fighter's nickname, may be empty string
            height      : str -- e.g. "6' 1"" stored raw from CSV
            weight      : str -- e.g. "185 lbs"
            reach       : str -- e.g. "74.0""
            stance      : str -- "Orthodox", "Southpaw", or "Switch"
            dob         : str -- date of birth e.g. "Jul / 14 / 1987"

        Optional args (computed by loader.py, default to 0):
            wins, losses, draws : int   -- career record
            slpm, str_acc, sapm, str_def : float -- striking stats
            td_avg, td_acc, td_def, sub_avg  : float -- grappling stats
        """

        # --- Identity ---

        self.first_name = first
        self.last_name = last
        self.nickname = nickname
        self.full_name = f"{first} {last}"

        # --- Physical Attributes ---
        #stored as raw strings exactly as they appear in the csv.
        #loader.py parses it

        self.height = height
        self.weight = weight
        self.reach = reach
        self.stance = stance
        self.dob = dob

        # --- Win/Loss/Draw Record ---
        # Integers. loader.py computes these by aggregating ufc_fight_results.csv.
        # default to 0 so a fighter can be constructed before data is available.

        self.wins = wins
        self.losses = losses
        self.draws = draws


        # --- Striking statistics ---
        self.slpm = slpm  # Significant Strikes Landed per Minute
        self.str_acc = str_acc  # Striking Accuracy
        self.sapm = sapm  # Significant Strikes Absorbed per Minute
        self.str_def = str_def  # Strike Defense

        # --- Grappling Statistics ---
        self.td_avg = td_avg  # Takedown Average per 15 min
        self.td_acc = td_acc  # Takedown Accuracy
        self.td_def = td_def  # Takedown Defense
        self.sub_avg = sub_avg  # Submission Attempt Average per 15 min


    def win_rate(self):
        """
        Returns the fighter's win rate as a float between 0.0 and 1.0.
        Returns 0.0 if the fighter has no recorded fights to avoid ZeroDivisionError.
        """
        total_fights = self.wins + self.losses + self.draws

        if total_fights == 0:
            return 0.0

        return self.wins / total_fights

    def __str__(self):
        """
        Returns a clean, human-readable summary of this fighter.
        Called automatically by print() and str().
        """
        nickname_display = f'"{self.nickname}"' if self.nickname else "No Nickname"

        return (
            f"--- {self.full_name} ({nickname_display}) ---\n"
            f"Physical:  Height: {self.height} | Weight: {self.weight} | "
            f"Reach: {self.reach} | Stance: {self.stance} | DOB: {self.dob}\n"
            f"Record:    {self.wins}W - {self.losses}L - {self.draws}D | "
            f"Win Rate: {self.win_rate():.1%}\n"
            f"Striking:  SLpM: {self.slpm} | Acc: {self.str_acc} | "
            f"SApM: {self.sapm} | Def: {self.str_def}\n"
            f"Grappling: TD Avg: {self.td_avg} | TD Acc: {self.td_acc} | "
            f"TD Def: {self.td_def} | Sub Avg: {self.sub_avg}"
        )





