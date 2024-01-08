import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.data_generation import load_realworld_data, sample_realworld_methylation_values, \
    estimate_beta_distribution_parameters, determine_correlation_matrix, generate_correlated_gaussian_data, \
    generate_bin_correlation_ranges, synthesize_correlated_gaussian_bins

MODULE_PATH = 'scripts.analysis.data_generation'


@pytest.fixture
def mock_read_hdf(mocker):
    return mocker.patch(f'{MODULE_PATH}.pd.read_hdf', return_value=pd.DataFrame({'A': [1, 2], 'B': [3, 4]}))


@pytest.fixture
def mock_realworld_data():
    return pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]})


@pytest.fixture
def mock_methyl_beta_values():
    return np.array(([0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3, 0.4]))


@pytest.fixture
def mock_uncorrelated_vector():
    return np.array(
        [0.25946043, -0.13786632, -2.15991597, -1.30200354, -0.84173935, -0.83198597, -0.58998776, 1.93719029,
         -0.54840062, -1.07933452])


@pytest.fixture
def mock_correlation_coefficients():
    return np.array([(0.1, 0.2), (0.3, 0.4)])


@pytest.fixture
def mock_beta_dist_fit(mocker):
    return mocker.patch(f'{MODULE_PATH}.beta_dist.fit', return_value=(0.6, 0.7, mocker.MagicMock, mocker.MagicMock))


@pytest.fixture
def mock_spearmanr(mocker):
    mock_result = type('MockResult', (),
                       {'statistic': np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])})
    return mocker.patch(f'{MODULE_PATH}.spearmanr', return_value=mock_result)


@pytest.fixture
def mock_generate_bin_correlation_ranges(mocker):
    return mocker.patch(f'{MODULE_PATH}.generate_bin_correlation_ranges', return_value=[[0.1, 0.2], [0.3, 0.4], [0.1, 0.2]])


@pytest.fixture
def mock_generate_correlated_gaussian_data(mocker):
    return mocker.patch(f'{MODULE_PATH}.generate_correlated_gaussian_data', return_value=np.random.normal(size=100))

def test_load_realworld_data(mock_realworld_data):
    with tempfile.NamedTemporaryFile(suffix='.h5', mode='wb', delete=False) as tmp:
        mock_realworld_data.to_hdf(tmp.name, key='test_data', mode='w')
        loaded_data = load_realworld_data(tmp.name)
        assert loaded_data.equals(mock_realworld_data)
    os.remove(tmp.name)


def test_load_realworld_data_with_mock(mock_read_hdf):
    result = load_realworld_data('dummy_path.h5')

    assert isinstance(result, pd.DataFrame)
    assert mock_read_hdf.call_count == 1


def test_load_realworld_data_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_realworld_data('invalid_path.h5')


def test_load_realworld_data_invalid_extension():
    with pytest.raises(ValueError):
        load_realworld_data('invalid_extension.txt')


def test_sample_realworld_methylation_values(mock_realworld_data):
    result = sample_realworld_methylation_values(2, mock_realworld_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (mock_realworld_data.shape[0], 2)


def test_sample_realworld_methylation_values_invalid(mock_realworld_data):
    with pytest.raises(ValueError):
        sample_realworld_methylation_values(4, mock_realworld_data)


def test_estimate_beta_distribution_parameters(mock_methyl_beta_values):
    alpha_params, beta_params = estimate_beta_distribution_parameters(mock_methyl_beta_values)

    assert alpha_params.shape == (4,) and beta_params.shape == (4,)
    assert (alpha_params > 0).all() and (beta_params > 0).all()


def test_estimate_beta_distribution_parameters_with_mock(mock_beta_dist_fit, mock_methyl_beta_values):
    alpha_params, beta_params = estimate_beta_distribution_parameters(mock_methyl_beta_values)

    assert alpha_params.shape == (4,) and beta_params.shape == (4,)
    assert (alpha_params > 0).all() and (beta_params > 0).all()
    assert mock_beta_dist_fit.call_count == mock_methyl_beta_values.shape[1]


def test_determine_correlation_matrix(mock_methyl_beta_values):
    correlation_matrix = determine_correlation_matrix(mock_methyl_beta_values)

    assert correlation_matrix.shape == (4, 4)
    assert np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-7)  # Check for symmetry


def test_determine_correlation_matrix_with_mock(mock_spearmanr, mock_methyl_beta_values):
    correlation_matrix = determine_correlation_matrix(mock_methyl_beta_values)

    assert correlation_matrix.shape == (4, 4)
    assert np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-7)  # Check for symmetry
    assert mock_spearmanr.call_count == 1


def test_generate_correlated_gaussian_data(mock_uncorrelated_vector):
    correlation_coefficient = 0.5
    result = generate_correlated_gaussian_data(mock_uncorrelated_vector, correlation_coefficient)
    result_correlation_coefficient = np.corrcoef(mock_uncorrelated_vector, result)[0, 1]

    assert len(result) == len(mock_uncorrelated_vector)
    assert np.isclose(result_correlation_coefficient, correlation_coefficient, atol=0.3), \
        f"Computed correlation({result_correlation_coefficient}) should be close to desired correlation({correlation_coefficient})"


@pytest.mark.parametrize("correlation_coefficient_distribution, n_bins, expected_results", [
    ([(-0.99, -0.70), (0.70, 0.85)], 4, [(-0.99, -0.70), (-0.99, -0.70), (0.70, 0.85), (0.70, 0.85)]),
    ([(-0.99, -0.70), (-0.99, -0.70), (0.70, 0.85), (0.70, 0.85)], 5, [(-0.99, -0.70), (-0.99, -0.70), (0.70, 0.85), (0.70, 0.85), (-0.99, -0.70)]),
    ([(-0.99, -0.70), (0.70, 0.85), (0.85, 0.99)], 5, [(-0.99, -0.7), (0.7, 0.85), (0.85, 0.99), (-0.99, -0.7), (0.7, 0.85)])])
def test_generate_bin_correlation_ranges(correlation_coefficient_distribution, n_bins, expected_results):
    actual_results = generate_bin_correlation_ranges(correlation_coefficient_distribution, n_bins)
    assert actual_results == expected_results


def test_synthesize_correlated_gaussian_bins(mock_correlation_coefficients):
    n_observations = 100
    n_sites = 10
    bin_size = 3
    n_bins = n_sites // bin_size
    result = synthesize_correlated_gaussian_bins(mock_correlation_coefficients, n_observations, n_sites, bin_size)
    result_mean = np.mean(result, axis=0)
    result_std = np.std(result, axis=0)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert np.allclose(result_mean, 0, atol=0.5)
    assert np.allclose(result_std, 1, atol=0.5)

    for i in range(n_bins):
        start_index = i * bin_size
        end_index = start_index + bin_size
        bin_data = result[:, start_index:end_index]
        min_corr, max_corr = mock_correlation_coefficients[i % len(mock_correlation_coefficients)]
        bin_corr_matrix = np.corrcoef(bin_data.T)
        # Check the correlation coefficients within each bin
        for j in range(bin_size):
            for k in range(j + 1, bin_size):
                corr_coef = bin_corr_matrix[j, k]
                closest_endpoint = min_corr if abs(corr_coef - min_corr) < abs(corr_coef - max_corr) else max_corr
                assert np.isclose(corr_coef, closest_endpoint, atol=0.3), (f"Correlation coefficient {corr_coef} not "
                                                                           f"close to range [{min_corr}, {max_corr}]")


def test_synthesize_correlated_gaussian_bins_with_mock(mock_correlation_coefficients, mock_generate_bin_correlation_ranges, mock_generate_correlated_gaussian_data):
    n_observations = 100
    n_sites = 10
    bin_size = 3
    n_bins = n_sites // bin_size

    result = synthesize_correlated_gaussian_bins(mock_correlation_coefficients, n_observations, n_sites, bin_size)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert mock_generate_bin_correlation_ranges.call_count == 1
    assert mock_generate_correlated_gaussian_data.call_count == (bin_size - 1) * n_bins


# def test_sample_legal_cvine_corrmat():
#     n_sites = 10
#     betaparam = 2.0
#     corr_matrix = sample_legal_cvine_corrmat(n_sites, betaparam)
#     assert np.allclose(corr_matrix, corr_matrix.T)
#     assert np.allclose(np.diag(corr_matrix), np.ones(n_sites))
#     assert np.allclose(corr_matrix.shape, (n_sites, n_sites))
#     assert np.all(np.linalg.eigvals(corr_matrix) >= 0)
#
#
# def test_synthesize_methyl_val_with_copula():
#     np.random.seed(123)
#     n_sites = 4
#     n_observations = 1000
#     alpha_params = np.random.uniform(low=0.5, high=2, size=n_sites)
#     beta_params = np.random.uniform(low=0.5, high=2, size=n_sites)
#     corr_matrix = np.array([[1.0, 0.3, 0.2, 0.1],
#                             [0.3, 1.0, 0.4, 0.2],
#                             [0.2, 0.4, 1.0, 0.3],
#                             [0.1, 0.2, 0.3, 1.0]])
#     synth_data = synthesize_methyl_val_with_copula_with_supplied_corrmat(correlation_matrix=corr_matrix,
#                                                                          n_observations=n_observations,
#                                                                          beta_dist_alpha_params=alpha_params,
#                                                                          beta_dist_beta_params=beta_params)
#     assert synth_data.shape == (n_observations, n_sites)
#     expected_means = [beta.mean(a=alpha_params[i], b=beta_params[i]) for i in range(n_sites)]
#     column_means = np.mean(synth_data, axis=0)
#     assert np.allclose(column_means, expected_means, rtol=0.1, atol=0.01)
#     corr_matrix_synth = np.corrcoef(synth_data.T)
#     assert np.allclose(corr_matrix_synth, corr_matrix, rtol=0.1, atol=0.05)
#
#
# def test_synthesize_methyl_val_with_autocorr():
#     np.random.seed(123)
#     n_sites = 5
#     n_observations = 200
#     alpha_params = np.random.uniform(low=0.5, high=2, size=n_sites)
#     beta_params = np.random.uniform(low=0.5, high=2, size=n_sites)
#     corr_coef_distribution = [(-0.99, -0.70), (0.70, 0.85)]
#     # corr_coef_distribution = [(0.6, 0.85)]
#     synth_data = synthesize_methyl_val_with_autocorr(corr_coef_distribution=corr_coef_distribution,
#                                                      n_observations=n_observations, n_sites=n_sites,
#                                                      beta_dist_alpha_params=alpha_params,
#                                                      beta_dist_beta_params=beta_params)
#     corr_mat = determine_correlation_matrix(synth_data)
#     print("-----------------------")
#     print(corr_mat)
#
#
# def test_synthesize_methyl_val_without_dependence():
#     n_sites = 10
#     n_observations = 100
#     beta_dist_alpha_params = np.array([2, 5, 1, 3, 4, 6, 5, 3, 2, 7])
#     beta_dist_beta_params = np.array([3, 2, 2, 4, 3, 6, 4, 2, 5, 6])
#     np.random.seed(123)
#     synth_beta_values = synthesize_methyl_val_without_dependence(n_sites, n_observations,
#                                                                  beta_dist_alpha_params,
#                                                                  beta_dist_beta_params)
#     assert type(synth_beta_values) == np.ndarray
#     assert synth_beta_values.shape == (n_observations, n_sites)
#     assert np.all((synth_beta_values >= 0) & (synth_beta_values <= 1))
#     column_means = synth_beta_values.mean(axis=0)
#     expected_means = beta_dist_alpha_params / (beta_dist_alpha_params + beta_dist_beta_params)
#     assert np.allclose(column_means, expected_means, rtol=0.1, atol=0.01)
#
#
# def test_beta_to_m():
#     with pytest.raises(ValueError):
#         beta_values = np.array([[0.2, 0.5, np.nan], [0.7, 0.4, 0.9]])
#         np.isnan(beta_to_m(beta_values)).any()
#         beta_values = np.array([[0.2, 0.5, np.inf], [0.7, 0.4, 0.9]])
#         assert np.isinf(beta_to_m(beta_values)).any()
#     beta_values = np.array([[0.2, 0.5, 0.1], [0.7, 0.4, 0.9]])
#     expected_output = np.array([[-2, 0, -3.169925], [1.22239242, -0.5849625, 3.169925]])
#     assert np.allclose(beta_to_m(beta_values), expected_output, rtol=1e-3)
#
#
# def test_simulate_methyl_data():
#     realworld_data = pd.DataFrame({'cpg1': [0.5, 0.3, 0.1], 'cpg2': [0.7, 0.6, 0.8], 'cpg3': [0.9, 0.85, 0.95]})
#     n_sites = 3
#     n_observations = 5
#     dependencies = False
#     result = simulate_methyl_data(realworld_data, n_sites, n_observations, dependencies)
#     assert isinstance(result, np.ndarray)
#     assert result.shape == (n_observations, n_sites)
#     assert np.all(result >= -34) and np.all(result <= 34)
#
#     realworld_data = load_realworld_data()
#     n_sites = 500
#     dependencies = True
#     result = simulate_methyl_data(realworld_data, n_sites, n_observations, dependencies)
#     assert isinstance(result, np.ndarray)
#     assert result.shape == (n_observations, n_sites)
#     assert np.all(result >= -34) and np.all(result <= 34)
#
#
# def test_generate_n_correlation_coefficients():
#     n_sites = 101
#     corr_coef_distribution = [(-0.99, -0.70), (0.70, 0.85)]
#     corr_coefs = generate_n_correlation_coefficients(corr_coef_distribution, n_sites)
#     assert len(corr_coefs) == 100
#
#

#

