import argparse
import ast
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregated_results', help='Path to the aggregated results file', required=True)
    parser.add_argument('--output_dir', help='Path to the output directory where the plots will be written',
                        required=True)
    parser.add_argument('--with_title', help='Path to the output directory where the plots will be written',
                        required=False, action='store_true')
    args = parser.parse_args()
    df = pd.read_csv(args.aggregated_results, sep="\t", header=0, index_col=False)
    df['reporting_histogram_bins'] = df['reporting_histogram_bins'].apply(ast.literal_eval)
    df['reporting_histogram'] = df['reporting_histogram'].apply(ast.literal_eval)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for index, row in df.iterrows():
        num_bins = len(row['reporting_histogram'])
        bin_edges = row['reporting_histogram_bins']
        custom_tick_labels = [bin_edges[0]]
        for i in range(1, num_bins): # [0, 1, 10, 100, 1000]
            if i == num_bins-1:
                custom_tick_labels.append(f'>= {bin_edges[i]}')
            else:
                custom_tick_labels.append(f'{bin_edges[i]}-{bin_edges[i + 1]-1}')
        custom_tick_labels = custom_tick_labels + [f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(1, num_bins)]
        hist_trace = go.Bar(x=custom_tick_labels, y=row['reporting_histogram'])
        config_cols = ["n_observations", "n_sites", "dependencies", "correlation_strength", "bin_size_ratio",
                       "statistical_test", "multipletest_correction_type", "alpha", "data_distribution"]
        config_dict = df.loc[index, config_cols].to_dict()
        if args.with_title is True:
            formatted_title = '<br>'.join([f'{key}: {value}' for key, value in config_dict.items()])
            layout = go.Layout(
                title={'text': formatted_title, 'y': 0.95},
                title_pad=dict(b=200),
                margin=dict(t=250),
                xaxis=dict(title='Number of false findings', tickvals=custom_tick_labels, ticktext=custom_tick_labels),
                yaxis=dict(title='Number of datasets')
            )
        else:
            layout = go.Layout(
                xaxis=dict(title='Number of false findings', tickvals=custom_tick_labels, ticktext=custom_tick_labels),
                yaxis=dict(title='Number of datasets')
            )

        fig = go.Figure(data=[hist_trace], layout=layout)
        fig.write_image(f"{args.output_dir}/{row['config_id']}.png", height=900, width=700)

