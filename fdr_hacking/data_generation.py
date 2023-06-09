import numpy as np
import pandas as pd
import os.path
from scipy.stats import norm, multivariate_normal as mvn, beta as beta_dist, spearmanr, beta


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


def sample_legal_cvine_corrmat(n_sites: int, betaparam: float) -> np.ndarray:
    """
    :param n_sites: The number of features to be included in the simulated correlation coefficient matrix
    (returns a n_sites x n_sites matrix). Note that if n_sites is larger than 1000, this approach becomes very slow and
    thus in the current implementation, if n_sites is larger than 1000, an error is thrown.
    :param betaparam: A float representing both alpha and beta parameters of a beta distribution. As this value is larger,
    the off-diagonal elements of the generated correlation matrix have smaller variance clustered around zero
    (no correlation) and as this value is larger the off-diagonal elements contain more high correlations and the
    variance increases. When the value of this parameter is 1, the off-diagonal elements of the correlation matrix
    elements follow close to a normal distribution.
    :return: A simulated correlation coefficient matrix of shape n_sites x n_sites
    """
    assert n_sites <= 1000, "Warning: This approach can be very slow for n_sites > 1000; consider combining " \
                            "multiple smaller chunks"
    P = np.zeros((n_sites, n_sites))
    S = np.eye(n_sites)
    for k in range(1, n_sites):
        for i in range(k + 1, n_sites):
            P[k, i] = np.random.beta(betaparam, betaparam)
            P[k, i] = (P[k, i] - 0.5) * 2
            p = P[k, i]
            for l in range(k - 1, -1, -1):
                p = p * np.sqrt((1 - P[l, i] ** 2) * (1 - P[l, k] ** 2)) + P[l, i] * P[l, k]
            S[k, i] = p
            S[i, k] = p
    permutation = np.random.permutation(n_sites)
    S = S[np.ix_(permutation, permutation)]
    S += 1e-6 * np.eye(n_sites)
    return S


def determine_correlation_matrix(methyl_beta_values: np.ndarray) -> np.ndarray:
    """
    Given a methylation dataset of beta values with features in columns of a ndimensional numpy array, this function
    computes the spearman rank correlation coefficients between each feature and returns a matrix of correlation
    coefficients as ndimensional numpy array.

    :param methyl_beta_values: A ndimensional numpy array containing methylation beta values, with features in columns
    :return: A ndimensional numpy array of pairwise correlation coefficients between each feature
    """
    return spearmanr(methyl_beta_values).statistic


def synthesize_methyl_val_with_copula(correlation_matrix: np.ndarray,
                                      n_observations: int,
                                      beta_dist_alpha_params: np.array,
                                      beta_dist_beta_params: np.array) -> np.ndarray:
    """
    :param correlation_matrix: A correlation coefficient matrix, where either the number of rows or columns represent
    the number of features to be included in the simulated methylation dataset
    :param n_observations: The number of observations to be included in the simulated methylation dataset
    :param beta_dist_alpha_params: A numpy array containing alpha parameters of beta distribution. Expected size of
    the numpy array is as many features as desired in the simulated methylation dataset.
    :param beta_dist_beta_params: A numpy array containing beta parameters of beta distribution. Expected size of
    the numpy array is as many features as desired in the simulated methylation dataset.
    :return: A simulated methylation dataset of beta values (as ndimensional numpy array) with desired
    correlation structure with n_observations (rows) x n_sites (columns).
    """
    assert len(beta_dist_alpha_params) == correlation_matrix.shape[0], "The length of alpha params array should match" \
                                                                       "correlation matrix shape"
    assert len(beta_dist_beta_params) == correlation_matrix.shape[0], "The length of beta params array should match" \
                                                                      "correlation matrix shape"
    dependency_structure = mvn(mean=np.zeros(correlation_matrix.shape[0]), cov=correlation_matrix, allow_singular=True)
    random_variables = dependency_structure.rvs(size=n_observations)
    uniform_random_variables = [norm.cdf(random_variables[:, i]) for i in range(random_variables.shape[1])]
    synth_beta_values = np.zeros((random_variables.shape[0], random_variables.shape[1]))
    for i in range(len(beta_dist_alpha_params)):
        synth_beta_values[:, i] = beta_dist(a=beta_dist_alpha_params[i], b=beta_dist_beta_params[i]).ppf(
            uniform_random_variables[i])
    return synth_beta_values


def generate_correlated_gaussian(x, corr):
    cov = np.array([[1.0, corr], [corr, 1.0]])
    L = np.linalg.cholesky(cov)
    uncorrelated_samples = np.random.normal(0, 1, len(x))
    correlated_samples = np.dot(L, [x, uncorrelated_samples])
    correlated_array = correlated_samples[1]
    return correlated_array


def generate_n_correlation_coefficients(corr_coef_distribution, n_sites):
    num_tuples = len(corr_coef_distribution)
    sites_per_tuple = n_sites // num_tuples
    coefficients = []
    for corr_range in corr_coef_distribution:
        min_corr, max_corr = corr_range
        tuple_coefficients = np.random.uniform(min_corr, max_corr, sites_per_tuple)
        coefficients.extend(tuple_coefficients)
    remaining_sites = n_sites % num_tuples
    for i in range(remaining_sites):
        min_corr, max_corr = corr_coef_distribution[i]
        coefficient = np.random.uniform(min_corr, max_corr)
        coefficients.append(coefficient)
    return coefficients


def synthesize_autoregressive_gaussian_variables(corr_coef_distribution: list, n_observations: int, n_sites: int):
    corr_coefs = generate_n_correlation_coefficients(corr_coef_distribution, n_sites)
    gaussian_vars_mat = np.zeros((n_observations, n_sites))
    for i, cor_coeff in enumerate(corr_coefs):
        if i < 1:
            gaussian_vars_mat[:, i] = np.random.normal(size=n_observations)
        else:
            gaussian_vars_mat[:, i] = generate_correlated_gaussian(gaussian_vars_mat[:, i - 1], cor_coeff)
    return gaussian_vars_mat


def transform_gaussian_to_beta(gaussian_vars_mat, beta_dist_alpha_params: np.array, beta_dist_beta_params: np.array):
    synth_beta_values = np.zeros((gaussian_vars_mat.shape[0], gaussian_vars_mat.shape[1]))
    uniform_random_variables = [norm.cdf(gaussian_vars_mat[:, i]) for i in range(gaussian_vars_mat.shape[1])]
    for i in range(len(beta_dist_alpha_params)):
        synth_beta_values[:, i] = beta_dist(a=beta_dist_alpha_params[i], b=beta_dist_beta_params[i]).ppf(
            uniform_random_variables[i])
    return synth_beta_values


def synthesize_methyl_val_with_autocorr(corr_coef_distribution: list, n_observations: int, n_sites: int,
                                        beta_dist_alpha_params: np.array,
                                        beta_dist_beta_params: np.array) -> np.ndarray:
    gaussian_vars_mat = synthesize_autoregressive_gaussian_variables(corr_coef_distribution, n_observations, n_sites)
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
                         dependencies: bool) -> np.ndarray:
    real_data = sample_realworld_methyl_val(n_sites=n_sites, realworld_data=realworld_data)
    alpha_params, beta_params = estimate_beta_dist_parameters(methyl_beta_values=real_data)
    if dependencies:
        assert n_sites % 500 == 0, "n_sites must be divisible by 500"
        num_chunks = n_sites // 500
        alpha_chunks = np.array_split(alpha_params, num_chunks)
        beta_chunks = np.array_split(beta_params, num_chunks)
        synth_datasets = []
        for i in range(num_chunks):
            corr_matrix = sample_legal_cvine_corrmat(n_sites=500, betaparam=0.5)
            alpha_chunk = alpha_chunks[i]
            beta_chunk = beta_chunks[i]
            synth_beta_values = synthesize_methyl_val_with_copula(correlation_matrix=corr_matrix,
                                                                  n_observations=n_observations,
                                                                  beta_dist_alpha_params=alpha_chunk,
                                                                  beta_dist_beta_params=beta_chunk)
            synth_datasets.append(synth_beta_values)
        synth_beta_values = np.concatenate(synth_datasets, axis=1)
    else:
        synth_beta_values = synthesize_methyl_val_without_dependence(n_sites=n_sites, n_observations=n_observations,
                                                                     beta_dist_alpha_params=alpha_params,
                                                                     beta_dist_beta_params=beta_params)
    methyl_datamat = beta_to_m(methyl_beta_values=synth_beta_values)
    return methyl_datamat
