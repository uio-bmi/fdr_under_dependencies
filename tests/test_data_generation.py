import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta
import itertools
from fdr_hacking.data_generation import synthesize_methyl_val_without_dependence, synthesize_methyl_val_with_copula, \
    sample_corr_mat_given_dependence_structure, sample_corr_mat_given_distribution, beta_to_m, load_eg_realworld_data, \
    sample_realworld_methyl_val, estimate_realworld_corrcoef_distribution, simulate_methyl_data


# def test_sample_realworld_methyl_val():
#     assert False
#
#
# def test_estimate_beta_dist_parameters():
#     assert False
#
#
# def test_estimate_realworld_dependence_structure():
#     assert False


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
    np.fill_diagonal(corr_matrix, 10)
    coef_range_1 = len(
        np.where(np.logical_and(corr_matrix >= -0.7882352941176471, corr_matrix <= -0.083781512605042))[0])
    coef_range_2 = len(
        np.where(np.logical_and(corr_matrix >= 0.3165546218487396, corr_matrix <= 0.9492997198879553))[0])
    assert coef_range_1 == coef_range_2


# def test_determine_correlation_matrix():
#     assert False


def test_synthesize_methyl_val_with_copula():
    np.random.seed(123)
    n_sites = 4
    n_observations = 1000
    alpha_params = np.random.uniform(low=0.5, high=2, size=n_sites)
    beta_params = np.random.uniform(low=0.5, high=2, size=n_sites)
    corr_matrix = np.array([[1.0, 0.3, 0.2, 0.1],
                            [0.3, 1.0, 0.4, 0.2],
                            [0.2, 0.4, 1.0, 0.3],
                            [0.1, 0.2, 0.3, 1.0]])
    synth_data = synthesize_methyl_val_with_copula(correlation_matrix=corr_matrix,
                                                   n_observations=n_observations,
                                                   beta_dist_alpha_params=alpha_params,
                                                   beta_dist_beta_params=beta_params)
    assert synth_data.shape == (n_observations, n_sites)
    expected_means = [beta.mean(a=alpha_params[i], b=beta_params[i]) for i in range(n_sites)]
    column_means = np.mean(synth_data, axis=0)
    assert np.allclose(column_means, expected_means, rtol=0.1, atol=0.01)
    corr_matrix_synth = np.corrcoef(synth_data.T)
    assert np.allclose(corr_matrix_synth, corr_matrix, rtol=0.1, atol=0.05)


def test_synthesize_methyl_val_without_dependence():
    n_sites = 10
    n_observations = 100
    beta_dist_alpha_params = np.array([2, 5, 1, 3, 4, 6, 5, 3, 2, 7])
    beta_dist_beta_params = np.array([3, 2, 2, 4, 3, 6, 4, 2, 5, 6])
    np.random.seed(123)
    synth_beta_values = synthesize_methyl_val_without_dependence(n_sites, n_observations,
                                                                 beta_dist_alpha_params,
                                                                 beta_dist_beta_params)
    assert type(synth_beta_values) == np.ndarray
    assert synth_beta_values.shape == (n_observations, n_sites)
    assert np.all((synth_beta_values >= 0) & (synth_beta_values <= 1))
    column_means = synth_beta_values.mean(axis=0)
    expected_means = beta_dist_alpha_params / (beta_dist_alpha_params + beta_dist_beta_params)
    assert np.allclose(column_means, expected_means, rtol=0.1, atol=0.01)


def test_beta_to_m():
    with pytest.raises(ValueError):
        beta_values = np.array([[0.2, 0.5, np.nan], [0.7, 0.4, 0.9]])
        np.isnan(beta_to_m(beta_values)).any()
        beta_values = np.array([[0.2, 0.5, np.inf], [0.7, 0.4, 0.9]])
        assert np.isinf(beta_to_m(beta_values)).any()
    beta_values = np.array([[0.2, 0.5, 0.1], [0.7, 0.4, 0.9]])
    expected_output = np.array([[-2, 0, -3.169925], [1.22239242, -0.5849625, 3.169925]])
    assert np.allclose(beta_to_m(beta_values), expected_output, rtol=1e-3)


def test_load_eg_realworld_data():
    real_data_df = load_eg_realworld_data()
    assert real_data_df.shape == (35, 836691)


def test_estimate_realworld_corrcoef_distribution():
    real_data_df = load_eg_realworld_data()
    methyl_beta_values = sample_realworld_methyl_val(n_sites=100, realworld_data=real_data_df)
    corrcoef_distr = estimate_realworld_corrcoef_distribution(methyl_beta_values)
    assert len(corrcoef_distr) == 20


def test_simulate_methyl_data():
    realworld_data = pd.DataFrame({'cpg1': [0.5, 0.3, 0.1], 'cpg2': [0.7, 0.6, 0.8], 'cpg3': [0.9, 0.85, 0.95]})
    n_sites = 3
    n_observations = 5
    dependencies = False
    result = simulate_methyl_data(realworld_data, n_sites, n_observations, dependencies)
    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert np.all(result >= -34) and np.all(result <= 34)

    dependencies = True
    corr_coef_distribution = [(0.1, 0.4), (0.4, 0.7), (0.7, 1.0)]
    result = simulate_methyl_data(realworld_data, n_sites, n_observations, dependencies, corr_coef_distribution)
    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert np.all(result >= -34) and np.all(result <= 34)
