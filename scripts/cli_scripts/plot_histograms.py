import argparse
import ast
import numpy as np
import pandas as pd


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregated_results', help='Path to the aggregated results file', required=True)
    parser.add_argument('--output_dir', help='Path to the output directory where the plots will be written', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.aggregated_results, sep="\t", header=0, index_col=False)
    df['reporting_histogram_bins'] = df['reporting_histogram_bins'].apply(ast.literal_eval)
    df['reporting_histogram'] = df['reporting_histogram'].apply(ast.literal_eval)
    df['reporting_histogram'] = df['reporting_histogram'].apply(lambda x: np.array(x))
    df['reporting_histogram_bins'] = df['reporting_histogram_bins'].apply(lambda x: np.array(x))
    for index, row in df.iterrows():
        # plot the histogram using plotly
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Bar(x=row['reporting_histogram_bins'], y=row['reporting_histogram'])])
        fig.update_layout(title_text='Histogram of significant findings', xaxis_title_text='Number of significant findings',
                          yaxis_title_text='Frequency')

