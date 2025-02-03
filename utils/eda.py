import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to draw boxplot of all the cols given
def draw_boxplots(cols, data, filename):
    fig, axes = plt.subplots(nrows=int(len(cols)/3), ncols=3, figsize=(10, 14))
    axes = axes.flatten()
    handles, labels = None, None 
    for i, col in enumerate(cols):
        sns.boxplot(x=data[col], hue=data['quality'], ax=axes[i])
        if handles is None and labels is None:
            handles, labels = axes[i].get_legend_handles_labels()
        axes[i].legend_.remove()
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")
    fig.legend(handles, labels, title="Wine Quality", ncol=len(labels), loc="upper center",
                 frameon=True, fancybox=True, shadow=True, fontsize=16)
    plt.savefig(filename)
    plt.clf()

# Function to draw correlation clustermap of data
def draw_clustermap(data, filename):
    sns.clustermap(
        data.corr(),
        annot=True,
        fmt=".001g",
        cmap="YlGnBu",
        figsize=(8, 10),
        cbar_pos=None,
    )
    plt.savefig(filename)
    plt.clf()

# Function to remove outliers based on 99.5 percentile cut-off
def remove_outliers(cols, data):
    quantiles = {}
    for col in cols:
        quantiles[col] = float(data[col].quantile([0.995]).values[0])
    for col in cols:
        data = data[data[col] <= quantiles[col]]
    return data

# Function to get the names of all numerical columns
def get_cols(data):
    return data.select_dtypes(include=["number"]).columns[:-1]

# Function to generate figures based on raw or cleaned data
def get_figures(data, type, cols):
    outfile = f"figures/{type}_Clustermap.png"
    draw_clustermap(data, outfile)
    outfile = f"figures/{type}_Boxplots.png"
    draw_boxplots(cols, data, outfile)

# Main function to process the input CSV file 
def process_input(filename):
    wines = pd.read_csv(filename)
    cols = get_cols(wines)
    get_figures(wines, "Rawdata", cols)

    wines = wines.drop(["Id", "total sulfur dioxide", "fixed acidity"], axis=1)
    wines = wines.drop_duplicates()
    cols = get_cols(wines)
 
    wines = remove_outliers(cols, wines)
    get_figures(wines, "Cleandata", cols)
    wines.to_csv("./data/WinesCleaned.csv", index=False)

# Main function called
if __name__ == "__main__":
    process_input("./data/WineQT.csv")