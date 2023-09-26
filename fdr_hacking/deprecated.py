import itertools
import numpy as np
from fdr_hacking.data_generation import sample_realworld_methyl_val, load_eg_realworld_data
from fdr_hacking.util import find_high_corr_sites_distribution, estimate_realworld_corrcoef_distribution


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


def sample_corr_mat_given_dependence_structure(n_sites: int, corr_threshold: float,
                                               dependence_structure: list) -> np.ndarray:
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
            corr_mat[i, i:end_idx] = np.round(np.random.uniform(0, 0.1, size=end_idx - i), 2)
            corr_mat[i:end_idx, i] = corr_mat[i, i:end_idx]
            if (end_idx - (i + 1)) < num_high_corr_features:
                num_high_corr_features = min(num_high_corr_features, end_idx - (i + 1))
            num_high_corr_features = max(1, num_high_corr_features)
            high_corr_idx = np.random.choice(range(i + 1, end_idx), size=num_high_corr_features, replace=False)
            corr_values = np.round(np.random.uniform(corr_threshold, 1, size=num_high_corr_features), 2)
            corr_mat[i, high_corr_idx] = corr_values
            corr_mat[high_corr_idx, i] = corr_mat[i, high_corr_idx]
    corr_mat = np.round(corr_mat, 4)
    np.fill_diagonal(corr_mat, 1.0)
    corr_mat += 1e-6 * np.eye(n_sites)
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
        n_values = int(len(triu_indices[0]) / len(corr_coef_distribution))
        corr_coef_values = np.concatenate((corr_coef_values,
                                           np.round(np.random.uniform(corr_range[0], corr_range[1], n_values), 2)))
    np.random.shuffle(corr_coef_values)
    corr_matrix[triu_indices] = corr_coef_values
    corr_matrix = corr_matrix + corr_matrix.T
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix += 1e-6 * np.eye(n_sites)
    return corr_matrix


def test_sample_corr_mat_given_dependence_structure():
    np.random.seed(123)
    n_sites = 1000
    corr_threshold = 0.5
    dependence_structure = [(0.18171, 0.37070), (0.12410, 0.18171), (0.09030, 0.12410), (0.06710, 0.09030),
                            (0.04930, 0.06710), (0.03420, 0.04930), (0.02270, 0.03420), (0.01350, 0.02270),
                            (0.006, 0.01350), (0.0, 0.006)]
    corr_mat = sample_corr_mat_given_dependence_structure(n_sites, corr_threshold, dependence_structure)
    assert corr_mat.shape == (n_sites, n_sites)
    assert np.allclose(np.diag(corr_mat), np.ones(n_sites))
    assert np.allclose(corr_mat, np.transpose(corr_mat))
    upper_triangle = np.triu(corr_mat)
    high_corr_counts = np.sum(np.abs(upper_triangle) >= corr_threshold, axis=1)
    deciles = np.percentile(high_corr_counts / len(high_corr_counts), np.arange(0, 101, 10))
    dependence_struct_flat = list(itertools.chain(*dependence_structure))
    dependence_struct_flat = list(dict.fromkeys(dependence_struct_flat))
    deciles_dependence_struct = np.sort(np.array(dependence_struct_flat))
    assert np.allclose(deciles, deciles_dependence_struct, rtol=0.1, atol=0.01)
    # assert np.all(np.linalg.eigvals(corr_mat) >= 0)


def test_sample_corr_mat_given_distribution():
    n_sites = 100
    corr_coef_distribution = [(-0.7882352941176471, -0.083781512605042), (-0.083781512605042, 0.0), (0.0, 0.0),
                              (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.054621848739495805),
                              (0.054621848739495805, 0.1773669467787117), (0.1773669467787117, 0.3165546218487396),
                              (0.3165546218487396, 0.9492997198879553)]
    corr_matrix = sample_corr_mat_given_distribution(n_sites, corr_coef_distribution)
    assert np.allclose(corr_matrix, corr_matrix.T)
    assert np.allclose(np.diag(corr_matrix), np.ones(n_sites))
    assert np.allclose(corr_matrix.shape, (n_sites, n_sites))
    coef_range_1 = len(
        np.where(np.logical_and(corr_matrix >= -0.7882352941176471, corr_matrix <= -0.083781512605042))[0])
    coef_range_2 = len(
        np.where(np.logical_and(corr_matrix >= 0.3165546218487396, corr_matrix <= 0.9492997198879553))[0])
    assert coef_range_1 == coef_range_2
    # assert np.all(np.linalg.eigvals(corr_matrix) >= 0)


def test_estimate_realworld_corrcoef_distribution():
    real_data_df = load_eg_realworld_data()
    methyl_beta_values = sample_realworld_methyl_val(n_sites=100, realworld_data=real_data_df)
    corrcoef_distr = estimate_realworld_corrcoef_distribution(methyl_beta_values)
    assert len(corrcoef_distr) == 20
