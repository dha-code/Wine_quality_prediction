import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot

def check_qual(df, synthetic_data, metadata):
    # Function to check quality of the synthetic dataset
    # The scores are printed on the std output and the image is saved to the figures folder
    diagnostic = run_diagnostic(df, synthetic_data, metadata)
    quality_report = evaluate_quality(df, synthetic_data, metadata)

    fig = get_column_plot(
        real_data=df,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_name='quality'
    )    
    fig.write_image("./figures/Synthetic data - Quality dist.png")

def generate_synthetic_data(filename, num_samples, outfile):
    # Function to generate synthetic data given the original dataset
    df = pd.read_csv(filename)
    metadata = Metadata.detect_from_dataframe(
        data=df,
        table_name='Wines')
    
    model = TVAESynthesizer(metadata) 
    model.fit(df)
    synthetic_data = model.sample(num_samples)

    # Combine the original data with the newly generated synthetic data
    expanded_df = pd.concat([df,synthetic_data])
    expanded_df.to_csv(outfile, index=False)
    check_qual(df, synthetic_data, metadata)

# Main function
if __name__ == "__main__":
    generate_synthetic_data("./data/WinesCleaned.csv",1000, "./data/TVsynthetic_wine_data.csv")