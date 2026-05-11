# UFC Fight Predictor

A command-line Python application that predicts the outcome of a UFC fight
between any two fighters in the dataset, based on real historical stats.

## Features

- Loads fighter data from four CSV files covering stats, records, and physical attributes
- Accepts any two fighter names as input from the command line
- Scores each fighter across key performance metrics
- Predicts a winner with a confidence percentage
- Displays a clean, formatted breakdown of the prediction

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
│   └── display.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/arvinzargaran/ufc_predictor.git
cd ufc_predictor
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the program**

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

## Data Sources

The CSV files in the `data/` folder were sourced from a publicly available
UFC dataset on GitHub. The dataset includes fighter details, physical
attributes, fight results, and per-fight statistics.

## Tech Stack

- **Python 3** — core language
- **pandas** — CSV loading and data manipulation
- **colorama** — coloured terminal output