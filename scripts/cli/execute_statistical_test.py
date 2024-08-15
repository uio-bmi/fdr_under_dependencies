import argparse
import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind


"""
This script is not used in Snakemake workflow. It is used as a standalone, helper script to execute t-test
on multiple datasets in order to analyse p-values distribution.
"""


def parse_args():
    """
    This function parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to the dataset files in tsv format', required=True)
    parser.add_argument('--output', help='Path to the output directory, where all the results should be stored',
                        required=True)
    args = parser.parse_args()
    return args


def main(data_path, output):
    """
    This function executes the statistical test on multiple datasets.

    :param data_path: Path to the dataset files in tsv format
    :param output: Path to the output directory, where all the results should be stored
    :return: None
    """
    dataset_list = glob.glob(os.path.join(data_path, "*.tsv"))
    p_values_summary_dict = {}
    test_statistics_dict = {}
    p_values_dict = {}
    percentiles = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    for dataset in dataset_list:
        p_values, test_statistic = execute_statistical_test_on_single_dataset(dataset, output)
        p_values_dict[os.path.basename(dataset)] = p_values
        p_values_summary_dict[os.path.basename(dataset)] = pd.Series(p_values).describe(
            percentiles=percentiles)
        test_statistics_dict[os.path.basename(dataset)] = pd.Series(test_statistic).describe(
            percentiles=percentiles)
    p_values_summary_df = pd.DataFrame(p_values_summary_dict)
    p_values_df = pd.DataFrame(p_values_dict)

    fig = px.ecdf(p_values_df, x=p_values_df.columns)
    fig.write_image(os.path.join(output, "p_values_ecdf.png"))
    test_statistic_df = pd.DataFrame(test_statistics_dict)
    p_values_summary_df.round(2).to_csv(os.path.join(output, "p_values_summary_statistics.tsv"), sep="\t")
    test_statistic_df.round(2).to_csv(os.path.join(output, "test_statistic_summary_statistics.tsv"), sep="\t")


def execute_statistical_test_on_single_dataset(data_path: str, output: str):
    """
    This function executes the statistical test on a single dataset.

    :param data_path: Path to the dataset file in tsv format
    :param output: Path to the output directory, where all the results should be stored
    :return: None
    """
    data = np.loadtxt(data_path, delimiter="\t")
    output_files_prefix = os.path.splitext(os.path.basename(data_path))[0]
    n_observations = data.shape[0]
    group_size = n_observations // 2
    group1_indices = list(range(group_size))
    group2_indices = list(range(group_size, n_observations))
    group1_data = data[group1_indices]
    group2_data = data[group2_indices]

    p_values = np.zeros(data.shape[1])
    test_statistics = np.zeros(data.shape[1])
    for col in range(data.shape[1]):
        test_statistics[col], p_values[col] = ttest_ind(group1_data[:, col], group2_data[:, col])
    np.savetxt(os.path.join(output, output_files_prefix + "_test_statistic.tsv"), test_statistics, delimiter="\t")
    np.savetxt(os.path.join(output, output_files_prefix + "_p_values.tsv"), p_values, delimiter="\t")
    px.histogram(p_values).write_image(os.path.join(output, output_files_prefix + "_p_values_histogram.png"))
    px.histogram(test_statistics).write_image(
        os.path.join(output, output_files_prefix + "_test_statistic_histogram.png"))
    return p_values, test_statistics


def execute():
    """
    This function is executed when this file is run as a script.
    """
    args = parse_args()
    main(args.data_path, args.output)
