import argparse
import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to the dataset files in tsv format', required=True)
    parser.add_argument('--output', help='Path to the output directory, where all the results should be stored',
                        required=True)
    args = parser.parse_args()
    # gather a file list of all the tsv files in args.data_path using glob.glob
    dataset_list = glob.glob(os.path.join(args.data_path, "*.tsv"))
    # execute statistical test and hold the results in a dict where the dataset name is the key and the value is a tuple
    # make a ECDF plot of the p-values for each dataset in a single chart

    p_values_summary_dict = {}
    test_statistic_summary_dict = {}
    p_values_dict = {}
    for dataset in dataset_list:
        # results_dict[os.path.basename(dataset)] = execute_statistical_test_on_single_dataset(dataset, args.output)
        p_values, test_statistic = execute_statistical_test_on_single_dataset(dataset, args.output)
        p_values_dict[os.path.basename(dataset)] = p_values
        p_values_summary_dict[os.path.basename(dataset)] = pd.Series(p_values).describe(
            percentiles=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                         0.9, 0.95, 1])
        test_statistic_summary_dict[os.path.basename(dataset)] = pd.Series(test_statistic).describe(
            percentiles=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.8, 0.9, 0.95, 1])
    p_values_summary_df = pd.DataFrame(p_values_summary_dict)
    p_values_df = pd.DataFrame(p_values_dict)
    # make a ECDF plot of the p-values for each dataset in p_values_df in a single chart, where each dataset is a line
    # in the chart
    fig = px.ecdf(p_values_df, x=p_values_df.columns)
    fig.write_image(os.path.join(args.output, "p_values_ecdf.png"))
    test_statistic_df = pd.DataFrame(test_statistic_summary_dict)
    p_values_summary_df.round(2).to_csv(os.path.join(args.output, "p_values_summary_statistics.tsv"), sep="\t")
    test_statistic_df.round(2).to_csv(os.path.join(args.output, "test_statistic_summary_statistics.tsv"), sep="\t")


def execute_statistical_test_on_single_dataset(data_path: str, output: str):
    data = np.loadtxt(data_path, delimiter="\t")
    # take the filename of the input data file and use it as the name of the output directory
    output_files_prefix = os.path.basename(data_path).replace(".tsv", "")
    n_obs = data.shape[0]
    group_size = n_obs // 2

    group1_indices = list(range(group_size))
    group2_indices = list(range(group_size, n_obs))

    group1_data = data[group1_indices]
    group2_data = data[group2_indices]
    p_values = np.zeros(data.shape[1])
    test_statistic = np.zeros(data.shape[1])
    for col in range(data.shape[1]):
        test_statistic[col], p_values[col] = ttest_ind(group1_data[:, col], group2_data[:, col])
    np.savetxt(os.path.join(output, output_files_prefix + "_test_statistic.tsv"), test_statistic, delimiter="\t")
    np.savetxt(os.path.join(output, output_files_prefix + "_p_values.tsv"), p_values, delimiter="\t")
    px.histogram(p_values).write_image(os.path.join(output, output_files_prefix + "_p_values_histogram.png"))
    px.histogram(test_statistic).write_image(
        os.path.join(output, output_files_prefix + "_test_statistic_histogram.png"))
    return p_values, test_statistic
