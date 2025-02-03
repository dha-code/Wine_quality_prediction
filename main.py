from utils.model import run_nn
from utils.eda import process_input

input_filename = "./data/WineQT.csv"
output_filename = "./data/WinesCleaned.csv"

process_input(input_filename)
run_nn(output_filename)