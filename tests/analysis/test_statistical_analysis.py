import numpy as np
import pytest
from scripts.analysis.statistical_analysis import perform_t_test, perform_limma_test, perform_rank_sum_test, \
    perform_ks_test, get_p_values

MODULE_PATH = 'scripts.analysis.statistical_analysis'


@pytest.fixture
def mock_statistical_tests(mocker):
    mocks = {
        't-test': mocker.patch(f'{MODULE_PATH}.perform_t_test', return_value=np.array([0.5, 0.5, 0.5])),
        'limma': mocker.patch(f'{MODULE_PATH}.perform_limma_test', return_value=np.array([0.4, 0.4, 0.4])),
        'rank-sum': mocker.patch(f'{MODULE_PATH}.perform_rank_sum_test', return_value=np.array([0.3, 0.3, 0.3])),
        'ks-test': mocker.patch(f'{MODULE_PATH}.perform_ks_test', return_value=np.array([0.2, 0.2, 0.2])),
    }
    return mocks


@pytest.fixture
def mock_ttest_ind(mocker):
    return mocker.patch(f'{MODULE_PATH}.ttest_ind', return_value=(1.0, 1.0))


@pytest.fixture
def mock_ranksums(mocker):
    return mocker.patch(f'{MODULE_PATH}.ranksums', return_value=(1.0, 1.0))


@pytest.fixture
def mock_kstest(mocker):
    return mocker.patch(f'{MODULE_PATH}.kstest', return_value=(1.0, 1.0))


@pytest.fixture
def mock_r_functions(mocker):
    mock_r = mocker.patch(f'{MODULE_PATH}.robjects.r')
    mock_globalenv = mocker.patch(f'{MODULE_PATH}.robjects.globalenv')
    mock_intvector = mocker.patch(f'{MODULE_PATH}.robjects.IntVector')

    return mock_globalenv, mock_intvector, mock_r


@pytest.fixture
def sample_data():
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])


@pytest.fixture
def group1_indices():
    return [0, 1]


@pytest.fixture
def group2_indices():
    return [2, 3]


def test_perform_t_test(sample_data, group1_indices, group2_indices):
    p_values = perform_t_test(sample_data, group1_indices, group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


def test_perform_t_test_with_mock(mock_ttest_ind, sample_data, group1_indices, group2_indices):
    p_values = perform_t_test(sample_data, group1_indices, group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert (p_values == 1.0).all()
    assert mock_ttest_ind.call_count == sample_data.shape[1]


def test_perform_limma_test(sample_data, group1_indices, group2_indices):
    p_values = perform_limma_test(sample_data, group1_indices, group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


def test_perform_limma_test_with_mocks(mock_r_functions, sample_data, group1_indices, group2_indices):
    _, _, mock_r = mock_r_functions
    p_values = perform_limma_test(sample_data, group1_indices, group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert mock_r.call_count == 10


def test_perform_rank_sum_test(sample_data, group1_indices, group2_indices):
    p_values = perform_rank_sum_test(sample_data, group1_indices, group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


def test_perform_rank_sum_test_with_mock(mock_ranksums, sample_data, group1_indices, group2_indices):
    p_values = perform_rank_sum_test(sample_data, group1_indices, group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert (p_values == 1.0).all()
    assert mock_ranksums.call_count == sample_data.shape[1]


def test_perform_ks_test(sample_data):
    p_values = perform_ks_test(sample_data)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


def test_perform_ks_test_with_mock(mock_kstest, sample_data):
    p_values = perform_ks_test(sample_data)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert (p_values == 1.0).all()
    assert mock_kstest.call_count == sample_data.shape[1]


@pytest.mark.parametrize("test_type, expected_results", [
    ('t-test', np.array([0.5, 0.5, 0.5])),
    ('limma', np.array([0.4, 0.4, 0.4])),
    ('rank-sum', np.array([0.3, 0.3, 0.3])),
    ('ks-test', np.array([0.2, 0.2, 0.2])),
])
def test_get_p_values_valid_with_mocks(test_type, expected_results, mock_statistical_tests, sample_data, group1_indices, group2_indices):
    p_values = get_p_values(sample_data, group1_indices, group2_indices, test_type)
    assert mock_statistical_tests[test_type].call_count == 1
    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert np.array_equal(p_values, expected_results)


@pytest.mark.parametrize("test_type", ['t-test', 'limma', 'rank-sum', 'ks-test'])
def test_get_p_values_valid(test_type, sample_data, group1_indices, group2_indices):
    p_values = get_p_values(sample_data, group1_indices, group2_indices, test_type)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()

def test_get_p_values_invalid_type(sample_data, group1_indices, group2_indices):
    with pytest.raises(ValueError):
        get_p_values(sample_data, group1_indices, group2_indices, 'invalid-test')


if __name__ == '__main__':
    pytest.main([__file__])

# def test_adjust_pvalues():
#     p_values = np.array([0.01, 0.05, 0.1, 0.001, 0.03])
#     adjusted_pvalues = adjust_p_values(p_values, method='bonferroni')
#     expected_adjusted_pvalues = np.array([0.05, 0.25, 0.5, 0.005, 0.15])
#     assert np.allclose(adjusted_pvalues, expected_adjusted_pvalues)
#     p_values = np.array([0.01, 0.05, 0.1, 0.001, 0.03])
#     adjusted_pvalues = adjust_p_values(p_values, method='bh')
#     expected_adjusted_pvalues = np.array([0.025, 0.0625, 0.1, 0.005, 0.05])
#     assert np.allclose(adjusted_pvalues, expected_adjusted_pvalues)
#     with pytest.raises(ValueError):
#         adjust_p_values(p_values, method='invalid_method')
#
#
# def test_count_significant_pvalues():
#     alpha = 0.05
#     adjusted_pvalues = np.array([])
#     num_significant = count_significant_p_values(adjusted_pvalues, alpha)
#     assert num_significant == 0
#     adjusted_pvalues = np.array([0.3, 0.6, 0.8])
#     num_significant = count_significant_p_values(adjusted_pvalues, alpha)
#     assert num_significant == 0
#     adjusted_pvalues = np.array([0.04, 0.003, 0.5, 0.8])
#     num_significant = count_significant_p_values(adjusted_pvalues, alpha)
#     assert num_significant == 2
