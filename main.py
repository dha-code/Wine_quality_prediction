from utils.model import run_nn
from utils.eda import process_input
from utils.synthetic_data import generate_synthetic_data
import kaggle

kaggle.api.dataset_download_file('yasserh/wine-quality-dataset','WineQT.csv', path='./data')
wine_dataset = "./data/WineQT.csv"
input_to_nn = "./data/WinesCleaned.csv"
synthetic=False

if synthetic:
    num_of_new_samples = 1000
    synthetic_dataset = "./data/TVsynthetic_wine_data.csv"
    generate_synthetic_data(input_to_nn, num_of_new_samples, synthetic_dataset)
    input_to_nn = synthetic_dataset

process_input(wine_dataset)
run_nn(input_to_nn)