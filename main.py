from utils.model import run_nn
from utils.eda import process_input
from utils.synthetic_data import generate_synthetic_data
import kaggle
import argparse

# Get the user input if they want to add synthetic data to the original dataset
parser = argparse.ArgumentParser(
    description="Train a neural network using the Kaggle Wine quality dataset, with an option to generate synthetic data."
)
parser.add_argument(
    "--synthetic",
    action="store_true",
    help="Enhance the wine quality dataset with optional 1K rows of synthetic data.",
)
args = parser.parse_args()

# Download the wine quality dataset from Kaggle
kaggle.api.dataset_download_file(
    "yasserh/wine-quality-dataset", "WineQT.csv", path="./data"
)
wine_dataset = "./data/WineQT.csv"

# Preprocess the data and clean it
process_input(wine_dataset)

# Use the cleaned data to run the run the NN 
input_to_nn = "./data/WinesCleaned.csv"

# Generate synthetic data if the user wishes
if args.synthetic:
    print("\nGenerating synthetic data based on the real dataset..\n")
    num_of_new_samples = 1000
    synthetic_dataset = "./data/TVsynthetic_wine_data.csv"
    generate_synthetic_data(input_to_nn, num_of_new_samples, synthetic_dataset)
    input_to_nn = synthetic_dataset

# Launch the neural network to create a wine prediction model and save the best model
print("Launching the Neural network...\n")
run_nn(input_to_nn)