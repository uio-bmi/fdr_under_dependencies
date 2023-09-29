import numpy as np
import pandas as pd
import os.path
import argparse
from scipy.stats import norm, beta as beta_dist, spearmanr


def load_eg_realworld_data():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'realworld_methyl_beta.h5')
    realworld_data = pd.read_hdf(file_path, "beta_values_df")
    return realworld_data


def sample_realworld_methyl_val(n_sites: int, realworld_data: pd.DataFrame) -> np.ndarray:
    """
    Given a pandas dataframe containing real-world methylation data with features in columns and observations in rows,
    this function returns a subset of the data by randomly sampling a subset of the features.

    :param n_sites: desired number of methylation sites in the sub-sampled data
    :param realworld_data: real-world methylation data in a pandas dataframe with features in columns
    :return: a sub-sampled real-world methylation data returned as ndimensional numpy array
    """
    assert n_sites <= realworld_data.shape[1], "desired n_sites larger than the number of sites in realworld_data"
    sampled_df = realworld_data.sample(n=n_sites, axis=1)
    return sampled_df.to_numpy()


def estimate_beta_dist_parameters(methyl_beta_values: np.ndarray) -> tuple:
    """
    Given a methylation dataset of beta values with features in columns of a ndimensional numpy array, this function
    estimates the parameters of the beta-distribution of each feature and returns a tuple containing two numpy arrays
    representing the alpha and beta parameters of a beta distribution. Note that methylation beta values are often
    modeled as following a beta distribution.

    :param methyl_beta_values: A ndimensional numpy array containing methylation beta values, with features in columns
    :return: A tuple containing two numpy arrays, representing alpha and beta params of beta distribution respectively
    """
    alpha_params = np.zeros(methyl_beta_values.shape[1])
    beta_params = np.zeros(methyl_beta_values.shape[1])
    for i in range(methyl_beta_values.shape[1]):
        alpha, beta, _, _ = beta_dist.fit(methyl_beta_values[:, i], floc=0, fscale=1)
        alpha_params[i] = alpha
        beta_params[i] = beta
    return alpha_params, beta_params


def determine_correlation_matrix(methyl_beta_values: np.ndarray) -> np.ndarray:
    """
    Given a methylation dataset of beta values with features in columns of a ndimensional numpy array, this function
    computes the spearman rank correlation coefficients between each feature and returns a matrix of correlation
    coefficients as ndimensional numpy array.

    :param methyl_beta_values: A ndimensional numpy array containing methylation beta values, with features in columns
    :return: A ndimensional numpy array of pairwise correlation coefficients between each feature
    """
    return spearmanr(methyl_beta_values).statistic


def generate_correlated_gaussian(x, corr):
    cov = np.array([[1.0, corr], [corr, 1.0]])
    L = np.linalg.cholesky(cov)
    uncorrelated_samples = np.random.normal(0, 1, len(x))
    correlated_samples = np.dot(L, [x, uncorrelated_samples])
    correlated_array = correlated_samples[1]
    return correlated_array


def generate_bin_correlation_ranges(corr_coef_distribution, n_bins):
    n_original_ranges = len(corr_coef_distribution)
    repetitions_per_range = n_bins // n_original_ranges
    remainder = n_bins % n_original_ranges
    new_corr_coef_distribution = []
    for each_range in corr_coef_distribution:
        new_corr_coef_distribution.extend([each_range] * repetitions_per_range)
    if remainder > 0:
        new_corr_coef_distribution.extend(corr_coef_distribution[:remainder])
    return new_corr_coef_distribution


def synthesize_correlated_gaussian_bins(corr_coef_distribution: list, n_observations: int, n_sites: int, bin_size: int):
    n_bins = n_sites // bin_size
    remainder = n_sites % bin_size
    bin_corr_ranges = generate_bin_correlation_ranges(corr_coef_distribution, n_bins)
    correlated_gaussian_bins = None
    for i in range(n_bins):
        correlated_gaussian_bin = np.zeros((n_observations, bin_size))
        correlated_gaussian_bin[:, 0] = np.random.normal(size=n_observations)
        for j in range(1, bin_size):
            min_corr, max_corr = bin_corr_ranges[i]
            correlated_gaussian_bin[:, j] = generate_correlated_gaussian(correlated_gaussian_bin[:, 0],
                                                                         np.random.uniform(min_corr, max_corr))
        if i == 0:
            correlated_gaussian_bins = correlated_gaussian_bin
        else:
            correlated_gaussian_bins = np.concatenate((correlated_gaussian_bins, correlated_gaussian_bin), axis=1)
    if remainder > 0:
        correlated_gaussian_bin = np.zeros((n_observations, remainder))
        for j in range(remainder):
            correlated_gaussian_bin[:, j] = np.random.normal(size=n_observations)
        correlated_gaussian_bins = np.concatenate((correlated_gaussian_bins, correlated_gaussian_bin), axis=1)
    return correlated_gaussian_bins


def transform_gaussian_to_beta(gaussian_vars_mat, beta_dist_alpha_params: np.array, beta_dist_beta_params: np.array):
    synth_beta_values = np.zeros((gaussian_vars_mat.shape[0], gaussian_vars_mat.shape[1]))
    uniform_random_variables = [norm.cdf(gaussian_vars_mat[:, i]) for i in range(gaussian_vars_mat.shape[1])]
    for i in range(len(beta_dist_alpha_params)):
        synth_beta_values[:, i] = beta_dist(a=beta_dist_alpha_params[i], b=beta_dist_beta_params[i]).ppf(
            uniform_random_variables[i])
    return synth_beta_values


def synthesize_methyl_val_with_correlated_bins(corr_coef_distribution: list, n_observations: int, n_sites: int,
                                               beta_dist_alpha_params: np.array, beta_dist_beta_params: np.array,
                                               bin_size: int) -> np.ndarray:
    gaussian_vars_mat = synthesize_correlated_gaussian_bins(corr_coef_distribution=corr_coef_distribution,
                                                            n_observations=n_observations,
                                                            n_sites=n_sites, bin_size=bin_size)
    synth_beta_values = transform_gaussian_to_beta(gaussian_vars_mat, beta_dist_alpha_params, beta_dist_beta_params)
    return synth_beta_values


def synthesize_methyl_val_without_dependence(n_sites: int, n_observations: int,
                                             beta_dist_alpha_params: np.array,
                                             beta_dist_beta_params: np.array) -> np.ndarray:
    """
    :param n_sites: The number of features to be included in the simulated methylation dataset
    :param n_observations: The number of observations to be included in the simulated methylation dataset
    :param beta_dist_alpha_params: A numpy array containing alpha parameters of beta distribution. Expected size of
    the numpy array is as many features as desired in the simulated methylation dataset
    :param beta_dist_beta_params: A numpy array containing beta parameters of beta distribution. Expected size of
    the numpy array is as many features as desired in the simulated methylation dataset
    :return:
    """
    synth_beta_values = np.zeros((n_observations, n_sites))
    for i in range(n_sites):
        alpha = beta_dist_alpha_params[i]
        beta = beta_dist_beta_params[i]
        site_samples = np.random.beta(alpha, beta, size=n_observations)
        synth_beta_values[:, i] = site_samples
    return synth_beta_values


def beta_to_m(methyl_beta_values: np.ndarray) -> np.ndarray:
    """
    :param methyl_beta_values: A ndimensional numpy array containing methylation beta values, with features in columns
    :return: A ndimensional numpy array containing methylation M values (because of having desirable properties for
     statistical testing), with features in columns
    """
    if np.any(np.isnan(methyl_beta_values)) or np.any(methyl_beta_values < 0) or np.any(methyl_beta_values > 1):
        raise ValueError("Invalid beta values found.")
    m_values = np.log2(methyl_beta_values / (1 - methyl_beta_values))
    if np.any(np.isinf(m_values)):
        max_m_value = np.max(np.abs(m_values[~np.isinf(m_values)]))
        m_values[np.isinf(m_values)] = np.sign(m_values[np.isinf(m_values)]) * max_m_value
    return m_values


def simulate_methyl_data(realworld_data: pd.DataFrame, n_sites: int, n_observations: int,
                         dependencies: bool, bin_size: int, corr_coef_distribution: list) -> np.ndarray:
    real_data = sample_realworld_methyl_val(n_sites=n_sites, realworld_data=realworld_data)
    alpha_params, beta_params = estimate_beta_dist_parameters(methyl_beta_values=real_data)
    if dependencies:
        assert bin_size <= n_sites, "bin_size cannot be larger than n_sites"
        synth_beta_values = synthesize_methyl_val_with_correlated_bins(
            corr_coef_distribution=corr_coef_distribution,
            n_observations=n_observations,
            n_sites=n_sites, beta_dist_alpha_params=alpha_params, beta_dist_beta_params=beta_params,
            bin_size=bin_size)
    else:
        synth_beta_values = synthesize_methyl_val_without_dependence(n_sites=n_sites, n_observations=n_observations,
                                                                     beta_dist_alpha_params=alpha_params,
                                                                     beta_dist_beta_params=beta_params)
    methyl_datamat = beta_to_m(methyl_beta_values=synth_beta_values)
    return methyl_datamat
