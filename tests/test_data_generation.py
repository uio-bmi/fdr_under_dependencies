import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta
from scripts.analysis.data_generation import synthesize_methyl_val_without_dependence, \
    synthesize_methyl_val_with_copula_with_supplied_corrmat, \
    beta_to_m, load_realworld_data, simulate_methyl_data, sample_legal_cvine_corrmat, \
    synthesize_methyl_val_with_autocorr, determine_correlation_matrix, generate_n_correlation_coefficients, \
    generate_bin_correlation_ranges, synthesize_correlated_gaussian_bins


def test_sample_legal_cvine_corrmat():
    n_sites = 10
    betaparam = 2.0
    corr_matrix = sample_legal_cvine_corrmat(n_sites, betaparam)
    assert np.allclose(corr_matrix, corr_matrix.T)
    assert np.allclose(np.diag(corr_matrix), np.ones(n_sites))
    assert np.allclose(corr_matrix.shape, (n_sites, n_sites))
    assert np.all(np.linalg.eigvals(corr_matrix) >= 0)


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
    synth_data = synthesize_methyl_val_with_copula_with_supplied_corrmat(correlation_matrix=corr_matrix,
                                                                         n_observations=n_observations,
                                                                         beta_dist_alpha_params=alpha_params,
                                                                         beta_dist_beta_params=beta_params)
    assert synth_data.shape == (n_observations, n_sites)
    expected_means = [beta.mean(a=alpha_params[i], b=beta_params[i]) for i in range(n_sites)]
    column_means = np.mean(synth_data, axis=0)
    assert np.allclose(column_means, expected_means, rtol=0.1, atol=0.01)
    corr_matrix_synth = np.corrcoef(synth_data.T)
    assert np.allclose(corr_matrix_synth, corr_matrix, rtol=0.1, atol=0.05)


def test_synthesize_methyl_val_with_autocorr():
    np.random.seed(123)
    n_sites = 5
    n_observations = 200
    alpha_params = np.random.uniform(low=0.5, high=2, size=n_sites)
    beta_params = np.random.uniform(low=0.5, high=2, size=n_sites)
    corr_coef_distribution = [(-0.99, -0.70), (0.70, 0.85)]
    # corr_coef_distribution = [(0.6, 0.85)]
    synth_data = synthesize_methyl_val_with_autocorr(corr_coef_distribution=corr_coef_distribution,
                                                     n_observations=n_observations, n_sites=n_sites,
                                                     beta_dist_alpha_params=alpha_params,
                                                     beta_dist_beta_params=beta_params)
    corr_mat = determine_correlation_matrix(synth_data)
    print("-----------------------")
    print(corr_mat)


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
    real_data_df = load_realworld_data()
    assert real_data_df.shape == (35, 836691)


def test_simulate_methyl_data():
    realworld_data = pd.DataFrame({'cpg1': [0.5, 0.3, 0.1], 'cpg2': [0.7, 0.6, 0.8], 'cpg3': [0.9, 0.85, 0.95]})
    n_sites = 3
    n_observations = 5
    dependencies = False
    result = simulate_methyl_data(realworld_data, n_sites, n_observations, dependencies)
    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert np.all(result >= -34) and np.all(result <= 34)

    realworld_data = load_realworld_data()
    n_sites = 500
    dependencies = True
    result = simulate_methyl_data(realworld_data, n_sites, n_observations, dependencies)
    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert np.all(result >= -34) and np.all(result <= 34)


def test_generate_n_correlation_coefficients():
    n_sites = 101
    corr_coef_distribution = [(-0.99, -0.70), (0.70, 0.85)]
    corr_coefs = generate_n_correlation_coefficients(corr_coef_distribution, n_sites)
    assert len(corr_coefs) == 100


def test_generate_bin_correlation_ranges():
    corr_coef_distribution = [(-0.99, -0.70), (0.70, 0.85)]
    corr_coef_ranges = generate_bin_correlation_ranges(corr_coef_distribution, 4)
    assert corr_coef_ranges == [(-0.99, -0.70), (-0.99, -0.70), (0.70, 0.85), (0.70, 0.85)]
    corr_coef_ranges = generate_bin_correlation_ranges(corr_coef_distribution, 5)
    assert corr_coef_ranges == [(-0.99, -0.70), (-0.99, -0.70), (0.70, 0.85), (0.70, 0.85), (-0.99, -0.70)]
    corr_coef_distribution = [(-0.99, -0.70), (0.70, 0.85), (0.85, 0.99)]
    corr_coef_ranges = generate_bin_correlation_ranges(corr_coef_distribution, 5)
    assert corr_coef_ranges == [(-0.99, -0.7), (0.7, 0.85), (0.85, 0.99), (-0.99, -0.7), (0.7, 0.85)]


def test_synthesize_correlated_gaussian_bins():
    n_sites = 10
    n_observations = 100
    corr_coef_distribution = [(-0.99, -0.70), (0.70, 0.85)]
    synth_data = synthesize_correlated_gaussian_bins(corr_coef_distribution, n_observations, n_sites, bin_size=5)
    assert synth_data.shape == (n_observations, n_sites)
    corr_mat = determine_correlation_matrix(synth_data)
    print("-----------------------")
    print(corr_mat)
