import os
import os
from src.loader import load_fighters
from src.predictor import predict
from src.display import display_prediction

def main():
     # build the path to the data folder
     data_dir = os.path.join(os.path.dirname(__file__), 'data')

     #load all fighters from the CSVs.
     fighters = load_fighters(data_dir)

     # Ask the user for two fighter names
     print("UFC Fight Predictor")
     print("-------------------")
     name1 = input("Enter fighter 1 name: ").strip().lower()
     name2 = input("Enter fighter 2 name: ").strip().lower()

     # Look up each fighter by name
     if name1 not in fighters:
         print(f"Fighter not found: '{name1}'")
         return

     if name2 not in fighters:
         print(f"Fighter not found: '{name2}'")
         return

     fighter1 = fighters[name1]
     fighter2 = fighters[name2]

     # Run the prediction and display the result
     result = predict(fighter1, fighter2)
     display_prediction(result)


if __name__ == "__main__":
    main()