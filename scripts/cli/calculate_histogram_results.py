import argparse
import ast

import numpy as np
import pandas as pd


def parse_args():
    """
    This function parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--concatenated_results', help='Path to the concatenated results file', required=True)
    parser.add_argument('--aggregated_results', help='Path to the aggregated results file', required=True)
    args = parser.parse_args()
    return args


def main(concatenated_results, aggregated_results):
    """
    This function aggregates the results of the statistical testing, calculates histogram values and saves the results.

    :param concatenated_results: Path to the concatenated results file
    :param aggregated_results: Path to the aggregated results file
    :return: None
    """
    df = pd.read_csv(concatenated_results, sep="\t", header=0, index_col=False)
    relevant_cols = ["n_observations", "n_sites", "dependencies", "correlation_strength", "bin_size_ratio",
                     "statistical_test", "multipletest_correction_type", "alpha", "data_distribution"]
    config_cols = [i for i in relevant_cols if i in df.columns]
    df['config_id'] = df[config_cols].apply(lambda row: '__'.join(f'{k}~{row[k]}' for k in config_cols), axis=1)

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
    hist_df.to_csv(aggregated_results, sep="\t", index=False)


def execute():
    """
    This function is executed when this file is run as a script.
    """
    args = parse_args()
    main(args.concatenated_results, args.aggregated_results)
