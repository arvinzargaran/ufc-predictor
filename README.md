# UFC Fight Predictor

A command-line Python application that predicts the outcome of a UFC fight
between any two fighters in the dataset, based on real historical stats.

## Features

- Loads fighter data from CSV files covering stats, records, and physical attributes
- Accepts any two fighter names as input from the command line
- Predicts a winner with a confidence percentage (trained ML ensemble)
- Predicts the method of victory (KO/TKO, submission, decision) for each fighter
- Predicts every fight on an upcoming UFC card straight from the schedule
- Refreshes its dataset from ufcstats.com with one command
- Displays a clean, formatted breakdown of the prediction

## Measured performance

Evaluated by training on the earliest 6,934 fights and testing on the most
recent 1,300 (2023-08-26 to 2026-06-27), so the model never sees a fight
before predicting it. Reproduce with `python3 -m src.evaluate`.

| | accuracy | log-loss | Brier | n |
|---|---|---|---|---|
| Model, stats only | 65.1% | 0.6359 | 0.2226 | 1300 |
| Market baseline (devigged close) | 70.0% | 0.5846 | 0.1996 | 1004 |
| Model + market blend | 69.7% | 0.5821 | 0.1988 | 1004 |
| Effective (blend where a line exists) | 67.5% | 0.6026 | 0.2081 | 1300 |

Read this honestly: **the betting market is the stronger predictor.** On the
fights where a closing line exists, the market picks more winners than the
model does. Blending the model into the market improves calibration by a
small margin (log-loss 0.5821 against 0.5846) while picking slightly fewer
winners, which is a real but modest edge on probability quality rather than
on raw accuracy. The stats-only model matters for the 296 fights that have
no line at all, where it is the only thing available.

Method of victory is a separate, harder problem: 53.7% against a 51.4%
baseline that always guesses "decision".

Backtesting also found that model/market disagreements were not historically
profitable. This is analysis, not betting advice.

## Project Structure

```
ufc_predictor/
├── data/
│   ├── ufc_fighter_details.csv
│   ├── ufc_fighter_tott.csv
│   ├── ufc_fight_results.csv
│   └── ufc_fight_stats.csv
├── src/
│   ├── __init__.py
│   ├── fighter.py
│   ├── loader.py
│   ├── predictor.py
│   ├── display.py
│   ├── model.py       # trained win + method-of-victory models
│   ├── backtest.py    # leakage-free chronological replay & features
│   ├── evaluate.py    # holdout evaluation (train past, test recent)
│   ├── scrape.py      # dataset refresh from ufcstats.com
│   ├── odds.py        # betting-line refresh from bestfightodds.com
│   ├── card.py        # predict a full upcoming card
│   ├── ledger.py      # log picks, grade them, track the live record
│   └── value.py       # compare model vs a betting line
├── main.py
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/arvinzargaran/ufc-predictor.git
cd ufc-predictor
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Fetch the dataset**

The CSVs are not committed (the `data/` folder is gitignored), so pull them
from ufcstats.com first, then train:

```bash
python3 -m src.scrape           # build data/ from ufcstats.com
python3 -m src.model            # train the model -> src/model.pkl
```

**4. Run the program**

```bash
python main.py
```

## Usage

Run the program and enter two fighter names when prompted:

```
$ python main.py
UFC Fight Predictor
-------------------
Enter fighter 1 name: jon jones
Enter fighter 2 name: stipe miocic
```

Fighter names are case-insensitive and should match the names in the dataset.

### Predict an upcoming card

```
python3 -m src.odds             # optional: pull current betting lines first
python3 -m src.card             # the next scheduled UFC event
python3 -m src.card --list      # see what's scheduled
python3 -m src.card mcgregor    # pick an event by name (or by number)
```

Prints a pick, win probability, and most likely method of victory for every
bout on the card. If `src.odds` has been run, picks show the market's
probability alongside and use the model's trained market blend; otherwise
they're stats-only.

### Refresh betting lines

```
python3 -m src.odds             # median moneyline across books, from bestfightodds.com
```

Upserts lines for upcoming cards into `data/ufc_odds.csv` (re-running
refreshes lines that moved). Fighter names and event dates are normalized
to match the ufcstats data automatically.

### Track the live record

Every `src.card` run logs its picks to `data/predictions.csv` (re-running
before fight night refreshes pending picks; graded picks are locked). After
an event, refresh results and grade:

```
python3 -m src.scrape           # pull the results
python3 -m src.ledger           # grade pending picks + show the record
```

The report shows pick accuracy, probability quality (log-loss), how the
betting market did on the same fights, the return of a hypothetical flat $1
bet on every lined pick, and how often the method call was right too. This
is the model's honest out-of-sample track record — the backtest can't lie
to you here.

### Refresh the dataset

```
python3 -m src.scrape           # pull new events/fights/fighters from ufcstats.com
python3 -m src.model            # then retrain so the model sees them
```

`--dry-run` shows what would be added without writing anything. Only the
ufcstats-sourced CSVs are refreshed; `ufc_odds.csv` comes from a separate
source, and fights without a betting line simply skip the market blend.

### Compare against a betting line

```
python3 -m src.value "max holloway" "dustin poirier" -150 +130
```

Shows the market's implied probabilities next to the model's, the size of any
disagreement, and the expected value of a $1 bet on each side. Note: in
backtesting, model/market disagreements were not historically profitable —
treat this as analysis, not betting advice.

### Retrain the model

```
python3 -m src.model
```

## Data Sources

The CSV files in the `data/` folder were sourced from a publicly available
UFC dataset on GitHub. The dataset includes fighter details, physical
attributes, fight results, and per-fight statistics.

## Tech Stack

- **Python 3** — core language
- **pandas** — CSV loading and data manipulation
- **scikit-learn** — logistic-regression ensemble and the market blend
- **tabulate** — formatted terminal tables