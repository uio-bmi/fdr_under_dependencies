import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.data_generation import load_realworld_data, sample_realworld_methylation_values, \
    estimate_beta_distribution_parameters, determine_correlation_matrix, generate_correlated_gaussian_data, \
    generate_bin_correlation_ranges, synthesize_correlated_gaussian_bins, transform_gaussian_to_beta, \
    synthesize_methyl_val_with_correlated_bins, synthesize_methyl_val_without_dependence, \
    synthesize_gaussian_dataset_without_dependence, beta_to_m, simulate_methyl_data

MODULE_PATH = 'scripts.analysis.data_generation'


@pytest.fixture
def mock_read_hdf(mocker):
    return mocker.patch(f'{MODULE_PATH}.pd.read_hdf', return_value=pd.DataFrame({'A': [1, 2], 'B': [3, 4]}))


@pytest.fixture
def mock_realworld_data():
    return pd.DataFrame({'A': [0.1, 0.2, 0.3, 0.4], 'B': [0.5, 0.6, 0.7, 0.8], 'C': [0.9, 0.10, 0.11, 0.12]})


@pytest.fixture
def mock_methyl_beta_values():
    return np.array(([0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.9, 0.4]))


@pytest.fixture
def mock_methyl_beta_values_with_border_values():
    return np.array(([0.6, 0.7, 0.8, 1.0], [0.0, 0.2, 0.9, 0.4]))


@pytest.fixture
def mock_gaussian_values():
    return np.array([[0.1, 0.2], [0.3, 0.4]])


@pytest.fixture
def mock_alpha_beta_params():
    return np.array([2.0, 3.0]), np.array([3.0, 2.0])


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
    return mocker.patch(f'{MODULE_PATH}.generate_bin_correlation_ranges',
                        return_value=[[0.1, 0.2], [0.3, 0.4], [0.1, 0.2]])


@pytest.fixture
def mock_generate_correlated_gaussian_data(mocker):
    return mocker.patch(f'{MODULE_PATH}.generate_correlated_gaussian_data', return_value=np.random.normal(size=100))


@pytest.fixture
def mock_synthesize_correlated_gaussian_bins(mocker):
    return mocker.patch(f'{MODULE_PATH}.synthesize_correlated_gaussian_bins', return_value=np.array([[-0.7, 1.3],
                                                                                                     [0.1, -0.9]]))


@pytest.fixture
def mock_transform_gaussian_to_beta(mocker):
    return mocker.patch(f'{MODULE_PATH}.transform_gaussian_to_beta', return_value=np.array([[0.7, 0.3],
                                                                                            [0.1, 0.4]]))


@pytest.fixture
def mock_sample_realworld_methylation_values(mocker):
    sampled_df = pd.DataFrame({'A': [0.1, 0.2, 0.3, 0.4], 'B': [0.5, 0.6, 0.7, 0.8]})
    return mocker.patch(f'{MODULE_PATH}.sample_realworld_methylation_values', return_value=sampled_df)


@pytest.fixture()
def mock_estimate_beta_distribution_parameters(mocker):
    return mocker.patch(f'{MODULE_PATH}.estimate_beta_distribution_parameters',
                        return_value=(np.array([2.0, 3.0, 1.0]), np.array([3.0, 2.0, 1.0])))


@pytest.fixture
def mock_synthesize_methyl_val_with_correlated_bins(mocker):
    return mocker.patch(f'{MODULE_PATH}.synthesize_methyl_val_with_correlated_bins',
                        return_value=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))


@pytest.fixture
def mock_synthesize_methyl_val_without_dependence(mocker):
    return mocker.patch(f'{MODULE_PATH}.synthesize_methyl_val_without_dependence',
                        return_value=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))


@pytest.fixture
def mock_beta_to_m(mocker):
    return mocker.patch(f'{MODULE_PATH}.beta_to_m', return_value=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))


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
    #assert (alpha_params > 0).all() and (beta_params > 0).all()


def test_estimate_beta_distribution_parameters_with_mock(mock_beta_dist_fit, mock_methyl_beta_values):
    alpha_params, beta_params = estimate_beta_distribution_parameters(mock_methyl_beta_values)

    assert alpha_params.shape == (4,) and beta_params.shape == (4,)
    assert (alpha_params > 0).all() and (beta_params > 0).all()
    assert mock_beta_dist_fit.call_count == mock_methyl_beta_values.shape[1]


def test_determine_correlation_matrix(mock_methyl_beta_values):
    result = determine_correlation_matrix(mock_methyl_beta_values)

    assert result.shape == (4, 4)
    assert np.allclose(result, result.T, atol=1e-7)  # Check for symmetry


def test_determine_correlation_matrix_with_mock(mock_spearmanr, mock_methyl_beta_values):
    result = determine_correlation_matrix(mock_methyl_beta_values)

    assert result.shape == (4, 4)
    assert np.allclose(result, result.T, atol=1e-7)  # Check for symmetry
    assert mock_spearmanr.call_count == 1


def test_generate_correlated_gaussian_data(mock_uncorrelated_vector):
    correlation_coefficient = 0.5
    result = generate_correlated_gaussian_data(mock_uncorrelated_vector, correlation_coefficient)
    result_correlation_coefficient = np.corrcoef(mock_uncorrelated_vector, result)[0, 1]

    assert len(result) == len(mock_uncorrelated_vector)
    assert np.isclose(result_correlation_coefficient, correlation_coefficient, atol=0.3), \
        f"Computed correlation({result_correlation_coefficient}) should be close to desired correlation({correlation_coefficient})"


@pytest.mark.parametrize("correlation_coefficient_distribution, n_bins, expected_result", [
    ([(-0.99, -0.70), (0.70, 0.85)], 4, [(-0.99, -0.70), (-0.99, -0.70), (0.70, 0.85), (0.70, 0.85)]),
    ([(-0.99, -0.70), (-0.99, -0.70), (0.70, 0.85), (0.70, 0.85)], 5,
     [(-0.99, -0.70), (-0.99, -0.70), (0.70, 0.85), (0.70, 0.85), (-0.99, -0.70)]),
    ([(-0.99, -0.70), (0.70, 0.85), (0.85, 0.99)], 5,
     [(-0.99, -0.7), (0.7, 0.85), (0.85, 0.99), (-0.99, -0.7), (0.7, 0.85)])])
def test_generate_bin_correlation_ranges(correlation_coefficient_distribution, n_bins, expected_result):
    result = generate_bin_correlation_ranges(correlation_coefficient_distribution, n_bins)
    assert result == expected_result


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


def test_synthesize_correlated_gaussian_bins_with_mock(mock_correlation_coefficients,
                                                       mock_generate_bin_correlation_ranges,
                                                       mock_generate_correlated_gaussian_data):
    n_observations = 100
    n_sites = 10
    bin_size = 3
    n_bins = n_sites // bin_size
    result = synthesize_correlated_gaussian_bins(mock_correlation_coefficients, n_observations, n_sites, bin_size)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert mock_generate_bin_correlation_ranges.call_count == 1
    assert mock_generate_correlated_gaussian_data.call_count == (bin_size - 1) * n_bins


def test_transform_gaussian_to_beta(mock_gaussian_values, mock_alpha_beta_params):
    alpha_params, beta_params = mock_alpha_beta_params
    result = transform_gaussian_to_beta(mock_gaussian_values, alpha_params, beta_params)

    assert result.shape == mock_gaussian_values.shape
    assert (result >= 0).all() & (result <= 1).all()
    expected_means = alpha_params / (alpha_params + beta_params)
    assert np.allclose(np.mean(result, axis=0), expected_means, atol=0.3)
    expected_stdevs = np.sqrt(
        alpha_params * beta_params / ((alpha_params + beta_params) ** 2 * (alpha_params + beta_params + 1)))
    assert np.allclose(np.std(result, axis=0), expected_stdevs, atol=0.3)


def test_synthesize_methyl_val_with_correlated_bins(mock_correlation_coefficients,
                                                    mock_alpha_beta_params):
    n_observations = 2
    n_sites = 2
    bin_size = 2
    beta_dist_alpha_params, beta_dist_beta_params = mock_alpha_beta_params
    result = synthesize_methyl_val_with_correlated_bins(
        mock_correlation_coefficients,
        n_observations, n_sites,
        beta_dist_alpha_params, beta_dist_beta_params,
        bin_size
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert (result >= 0).all() & (result <= 1).all()


def test_synthesize_methyl_val_with_correlated_bins_with_mock(mock_synthesize_correlated_gaussian_bins,
                                                              mock_transform_gaussian_to_beta,
                                                              mock_correlation_coefficients,
                                                              mock_alpha_beta_params):
    n_observations = 2
    n_sites = 2
    bin_size = 2
    beta_dist_alpha_params, beta_dist_beta_params = mock_alpha_beta_params
    _ = synthesize_methyl_val_with_correlated_bins(
        mock_correlation_coefficients,
        n_observations, n_sites,
        beta_dist_alpha_params, beta_dist_beta_params,
        bin_size
    )

    assert mock_synthesize_correlated_gaussian_bins.call_count == 1
    assert mock_transform_gaussian_to_beta.call_count == 1


def test_synthesize_methyl_val_without_dependence(mock_alpha_beta_params):
    n_sites = 2
    n_observations = 100
    beta_dist_alpha_params, beta_dist_beta_params = mock_alpha_beta_params
    np.random.seed(0)
    result = synthesize_methyl_val_without_dependence(n_sites, n_observations,
                                                      beta_dist_alpha_params,
                                                      beta_dist_beta_params)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert np.all((result >= 0) & (result <= 1))
    column_means = result.mean(axis=0)
    expected_means = beta_dist_alpha_params / (beta_dist_alpha_params + beta_dist_beta_params)
    assert np.allclose(column_means, expected_means, atol=0.3)


def test_synthesize_gaussian_dataset_without_dependence():
    n_sites = 5
    n_observations = 1000
    np.random.seed(0)
    result = synthesize_gaussian_dataset_without_dependence(n_sites, n_observations)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    mean = np.mean(result)
    std_dev = np.std(result)
    assert np.isclose(mean, 0, atol=0.1)
    assert np.isclose(std_dev, 1, atol=0.1)


def test_beta_to_m(mock_methyl_beta_values_with_border_values):
    result = beta_to_m(mock_methyl_beta_values_with_border_values)

    assert result.shape == mock_methyl_beta_values_with_border_values.shape
    assert not np.any(np.isinf(result))


@pytest.mark.parametrize("beta_values", [
    np.array([[0.2, 0.5, np.nan], [0.7, 0.4, 0.9]]),
    np.array([[0.2, 0.5, np.inf], [0.7, 0.4, 0.9]]),
    np.array([[1.2, 0.5, 2.0], [0.7, 0.4, 0.9]])
])
def test_beta_to_m_invalid_beta_values(beta_values):
    with pytest.raises(ValueError):
        beta_to_m(beta_values)


def test_simulate_methyl_data_invalid_bin_size(mock_realworld_data):
    with pytest.raises(ValueError):
        n_sites = 3
        n_observations = 4
        dependencies = True
        _ = simulate_methyl_data(mock_realworld_data, n_sites, n_observations, dependencies, bin_size=4)


def test_simulate_methyl_data(mock_realworld_data, mock_correlation_coefficients):
    n_sites = 3
    n_observations = 4
    dependencies = True
    bin_size = 2
    result = simulate_methyl_data(mock_realworld_data, n_sites, n_observations, dependencies, bin_size=bin_size,
                                  correlation_coefficient_distribution=mock_correlation_coefficients)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)


@pytest.mark.parametrize("dependencies, expected_synthesize_methyl_val_with_correlated_bins, "
                         "expected_synthesize_methyl_val_without_dependence", [(True, 1, 0), (False, 0, 1)])
def test_simulate_methyl_data_with_mock(mock_realworld_data, mock_correlation_coefficients, dependencies,
                                        mock_sample_realworld_methylation_values,
                                        mock_estimate_beta_distribution_parameters,
                                        mock_synthesize_methyl_val_with_correlated_bins,
                                        mock_synthesize_methyl_val_without_dependence, mock_beta_to_m,
                                        expected_synthesize_methyl_val_with_correlated_bins,
                                        expected_synthesize_methyl_val_without_dependence):
    n_sites = 3
    n_observations = 2
    bin_size = 2
    result = simulate_methyl_data(mock_realworld_data, n_sites, n_observations, dependencies, bin_size=bin_size,
                                  correlation_coefficient_distribution=mock_correlation_coefficients)

    assert isinstance(result, np.ndarray)
    assert result.shape == (n_observations, n_sites)
    assert mock_sample_realworld_methylation_values.call_count == 1
    assert mock_estimate_beta_distribution_parameters.call_count == 1
    assert mock_synthesize_methyl_val_with_correlated_bins.call_count == expected_synthesize_methyl_val_with_correlated_bins
    assert mock_synthesize_methyl_val_without_dependence.call_count == expected_synthesize_methyl_val_without_dependence
    assert mock_beta_to_m.call_count == 1


if __name__ == '__main__':
    pytest.main([__file__])
