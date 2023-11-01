import argparse
import ast
from fdr_hacking.data_generation import *
import numpy as np
import pandas as pd


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concatenated_results', help='Path to the concatenated results file', required=True)
    parser.add_argument('--aggregated_results', help='Path to the aggregated results file', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.concatenated_results, sep="\t", header=0, index_col=False)
    relevant_cols = ["n_observations", "n_sites", "dependencies", "correlation_strength", "bin_size_ratio",
                   "statistical_test", "multipletest_correction_type", "alpha", "data_distribution"]
    config_cols = [i for i in relevant_cols if i in df.columns]
    config_id_dict = df[config_cols].to_dict('records')
    df['config_id'] = ['__'.join(f'{k}~{v}' for k, v in i.items()) for i in config_id_dict]
    config_cols = config_cols + ["config_id", "reporting_histogram_bins"]
    df_grouped = df.groupby(config_cols)
    hist_df = pd.DataFrame(columns=config_cols + ["reporting_histogram"])
    for name, group in df_grouped:
        uniq_row = list(name)
        hist, bin_edges = np.histogram(a=group['num_significant_findings'],
                                       bins=ast.literal_eval(group['reporting_histogram_bins'].iloc[0]))
        uniq_row.append(list(hist))
        hist_df = pd.concat([hist_df, pd.DataFrame([uniq_row], columns=config_cols + ["reporting_histogram"])],
                            ignore_index=True)
    hist_df.to_csv(args.aggregated_results, sep="\t", index=False)

