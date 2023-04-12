import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal as mvn, beta as beta_dist, rankdata, spearmanr


def sample_realworld_methyl_val(n_sites: int, realworld_data: pd.DataFrame) -> np.ndarray:
    """
    Given a pandas dataframe containing real-world methylation data with features in columns and observations in rows,
    this function returns a subset of the data by randomly sampling a subset of the features.

    :param n_sites: desired number of methylation sites in the sub-sampled data
    :param realworld_data: real-world methylation data in a pandas dataframe with features in columns
    :return: a sub-sampled real-world methylation data returned as ndimensional numpy array
    """
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


def estimate_realworld_dependence_structure(n_sites: int, realworld_data: pd.DataFrame,
                                            corr_threshold: float, n_times: int) -> list:
    """
    Given a pandas dataframe containing real-world methylation data with features in columns and observations in rows,
    this function sub-samples a subset of features (specified by n_sites parameter). The sub-sampled data is used as
    a basis to find the deciles of counts of highly correlated features. The high correlation is defined by
    corr_threshold parameter. This process is repeated multiple times (specified by n_times parameter) and the average
    of deciles are computed. These deciles are then converted to discrete bins, which are returned as a list of tuples.
    These discrete intervals can be used as a basis to sample correlation coefficients when simulating a
    correlation coefficient matrix, thus reflecting the dependence structure of real-world methylation data.

    :param n_sites: Number of features to sub-sample from the supplied real-world methylation data
    :param realworld_data: real-world methylation data in a pandas dataframe with features in columns
    :param corr_threshold: A float indicating the absolute correlation coefficient between 0 and 1 (e.g. 0.4)
    to find "high" correlation features
    :param n_times: number of times to repeat for better estimation of dependence structure in real-world methylation data
    :return: A list of tuples, where each tuple represents a discrete interval that can be used when simulating
    correlation coefficient matrix
    """
    n_times_deciles = []
    for i in range(n_times):
        methyl_beta_values = sample_realworld_methyl_val(n_sites, realworld_data)
        n_times_deciles[i] = find_high_corr_sites_distribution(methyl_beta_values, corr_threshold)
    deciles_avg = np.mean(n_times_deciles, axis=0)
    intervals = [(deciles_avg[i] / n_sites, deciles_avg[i + 1] / n_sites) for i in range(len(deciles_avg) - 1)]
    intervals = list(reversed(intervals))
    return intervals


def sample_corr_mat_given_dependence_structure(n_sites: int, corr_threshold: float, dependence_structure: list) -> np.ndarray:
    """

    :param n_sites: The number of features to be included in the simulated correlation coefficient matrix
    (returns a n_sites x n_sites matrix).
    :param corr_threshold: A float indicating the absolute correlation coefficient between 0 and 1 (e.g. 0.4), which
    has to be used when simulating "high" correlations
    :param dependence_structure: A list of tuples, where each tuple represents a discrete interval that can be used
    when simulating correlation coefficient matrix. This has to be in a format as returned by
    `estimate_realworld_dependence_structure` function.
    :return: A simulated correlation coefficient matrix of shape n_sites x n_sites
    """
    corr_mat = np.zeros((n_sites, n_sites))
    for decile, percentile_range in enumerate(dependence_structure):
        start_idx = int(decile * n_sites / 10)
        end_idx = n_sites
        for i in range(start_idx, end_idx - 1):
            num_high_corr_features = int(np.random.uniform(*percentile_range) * n_sites)
            corr_mat[i, i:end_idx] = np.random.uniform(0, 0.1, size=end_idx - i)
            corr_mat[i:end_idx, i] = corr_mat[i, i:end_idx]
            if (end_idx - (i + 1)) < num_high_corr_features:
                num_high_corr_features = min(num_high_corr_features, end_idx - (i + 1))
            num_high_corr_features = max(1, num_high_corr_features)
            high_corr_idx = np.random.choice(range(i + 1, end_idx), size=num_high_corr_features, replace=False)
            corr_values = np.random.uniform(corr_threshold, 1, size=num_high_corr_features)
            corr_mat[i, high_corr_idx] = corr_values
            corr_mat[high_corr_idx, i] = corr_mat[i, high_corr_idx]
    corr_mat = np.round(corr_mat, 4)
    np.fill_diagonal(corr_mat, 1)
    return corr_mat


def sample_corr_mat_given_distribution(n_sites: int, corr_coef_distribution: list) -> np.ndarray:
    """

    :param n_sites: The number of features to be included in the simulated correlation coefficient matrix
    (returns a n_sites x n_sites matrix).
    :param corr_coef_distribution: A list of tuples, where each tuple represents a discrete interval that can be used
    when simulating correlation coefficient matrix. Unlike in `sample_corr_mat_given_dependence_structure`, each tuple
    in this list represent a range of correlation coefficients. Correlation coefficients will be drawn equally randomly
    from each tuple's range and filled in simulated correlation coefficient matrix randomly while respecting the
    symmetry of upper and lower triangles.
    :return: A simulated correlation coefficient matrix of shape n_sites x n_sites
    """
    corr_matrix = np.eye(n_sites)
    triu_indices = np.triu_indices(len(corr_matrix), k=1)
    corr_coef_values = np.array([])
    for corr_range in corr_coef_distribution:
        n_values = int(len(triu_indices[0])/len(corr_coef_distribution))
        corr_coef_values = np.concatenate((corr_coef_values, np.random.uniform(corr_range[0], corr_range[1], n_values)))
    np.random.shuffle(corr_coef_values)
    corr_matrix[triu_indices] = corr_coef_values
    corr_matrix = corr_matrix + corr_matrix.T
    np.fill_diagonal(corr_matrix, 1)
    return corr_matrix


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
    dependency_structure = mvn(mean=np.zeros(correlation_matrix.shape[0]), cov=correlation_matrix)
    random_variables = dependency_structure.rvs(size=n_observations)
    uniform_random_variables = [norm.cdf(random_variables[:, i]) for i in range(random_variables.shape[1])]
    synth_beta_values = np.zeros((random_variables.shape[0], random_variables.shape[1]))
    for i in range(len(beta_dist_alpha_params)):
        synth_beta_values[:, i] = beta_dist(a=beta_dist_alpha_params[i], b=beta_dist_beta_params[i]).ppf(
            uniform_random_variables[i])
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
