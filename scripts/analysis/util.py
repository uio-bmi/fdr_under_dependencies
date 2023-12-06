import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

from scripts.analysis.data_generation import determine_correlation_matrix


def find_high_corr_sites_distribution(methyl_beta_values: np.ndarray, corr_threshold: float) -> np.array:
    """
    Given a methylation dataset of beta values with features in columns of a ndimensional numpy array, this function
    computes the spearman rank correlation between the features and returns the deciles of the high correlation counts.
    The high correlation counts are determined based on a threshold parameter for absolute correlation coefficient.
    Only the upper triangle of the correlation coefficient matrix is used (excluding diagonal) when counting the
    number of highly correlated features. The returned deciles describe for e.g. the following: 10% of the total
    features are highly correlated with upto 30% of other features.

    :param methyl_beta_values: A ndimensional numpy array containing methylation beta values, with features in columns
    :param corr_threshold: A float indicating the absolute correlation coefficient between 0 and 1 (e.g. 0.4)
    :return: The deciles of counts of highly correlated features
    """
    corr_mat = determine_correlation_matrix(methyl_beta_values)
    corr_mat = np.triu(corr_mat, k=1)
    high_corr_counts = np.sum(np.abs(corr_mat) >= corr_threshold, axis=1)
    deciles = np.percentile(high_corr_counts, np.arange(0, 101, 10))
    return deciles


def estimate_realworld_corrcoef_distribution(methyl_beta_values: np.ndarray) -> list:
    """

    :param methyl_beta_values: A ndimensional numpy array containing methylation beta values, with features in columns
    :return: A list of tuples, where each tuple represents an interval of correlation coefficient range. The tuples
    are constructed based on empirical quantiles of correlation coefficients in one of the triangles of correlation
    coefficient matrix on a sample of real-world data.
    """
    corr_mat = determine_correlation_matrix(methyl_beta_values)
    corr_mat = np.triu(corr_mat, k=1)
    quantiles = np.percentile(corr_mat, np.arange(0, 101, 5))
    intervals = [(round(quantiles[i], 2), round(quantiles[i + 1], 2)) for i in range(len(quantiles) - 1)]
    return intervals


def plot_correlation_histogram(correlation_matrix):
    rows, cols = np.triu_indices(correlation_matrix.shape[0], k=1)
    off_diag = correlation_matrix[rows, cols]
    plt.hist(off_diag, bins='auto', alpha=0.7, rwidth=0.85)
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.title('Histogram of Off-Diagonal Correlation Elements')
    plt.grid(True)
    plt.show()


def parse_yaml_file(yaml_file_path: str) -> dict:
    with open(yaml_file_path, "r") as yaml_file:
        try:
            yaml_obj = yaml.load(yaml_file, Loader=yaml.FullLoader)
            assert yaml_obj is not None, "The supplied yaml file," + yaml_file_path + ", is empty"
        except Exception as e:
            print(
                "Error: that looks like an invalid yaml file. Consider validating your yaml file using one of the "
                "online yaml validators; for instance: https://jsonformatter.org/yaml-validator")
            print("Exception: %s" % str(e))
            sys.exit(1)
    return yaml_obj

