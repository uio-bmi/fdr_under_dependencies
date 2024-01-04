import numpy as np
import pytest
from scripts.analysis.statistical_analysis import perform_t_test, perform_limma_test, perform_rank_sum_test, \
    perform_ks_test, get_p_values, adjust_p_values, count_significant_p_values, quantify_significance

MODULE_PATH = 'scripts.analysis.statistical_analysis'


@pytest.fixture
def mock_sample_data():
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])


@pytest.fixture
def mock_group1_indices():
    return [0, 1]


@pytest.fixture
def mock_group2_indices():
    return [2, 3]


@pytest.fixture
def mock_p_values():
    return np.array([0.7, 0.7, 0.7])


@pytest.fixture
def mock_adjusted_p_values():
    return np.array([0.7, 0.5, 0.2])


@pytest.fixture
def mock_alpha():
    return 0.5


@pytest.fixture
def mock_test_type():
    return 't-test'


@pytest.fixture
def mock_correction_method():
    return 'bonferroni'


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
def mock_multipletests(mocker):
    return mocker.patch(f'{MODULE_PATH}.multipletests',
                        return_value=(mocker.MagicMock, np.array([0.5, 0.5, 0.5]), mocker.MagicMock, mocker.MagicMock))


@pytest.fixture
def mock_r_functions(mocker):
    mock_r = mocker.patch(f'{MODULE_PATH}.robjects.r')
    mock_globalenv = mocker.patch(f'{MODULE_PATH}.robjects.globalenv')
    mock_intvector = mocker.patch(f'{MODULE_PATH}.robjects.IntVector')

    return mock_globalenv, mock_intvector, mock_r


@pytest.fixture
def mock_get_p_values(mocker):
    return mocker.patch(f'{MODULE_PATH}.get_p_values', return_value=np.array([0.7, 0.7, 0.7]))


@pytest.fixture
def mock_adjust_p_values(mocker):
    return mocker.patch(f'{MODULE_PATH}.adjust_p_values', return_value=np.array([0.4, 0.4, 0.4]))


@pytest.fixture
def mock_count_significant_p_values(mocker):
    return mocker.patch(f'{MODULE_PATH}.count_significant_p_values', return_value=3)


def test_perform_t_test(mock_sample_data, mock_group1_indices, mock_group2_indices):
    p_values = perform_t_test(mock_sample_data, mock_group1_indices, mock_group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


def test_perform_t_test_with_mock(mock_ttest_ind, mock_sample_data, mock_group1_indices, mock_group2_indices):
    p_values = perform_t_test(mock_sample_data, mock_group1_indices, mock_group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert (p_values == 1.0).all()
    assert mock_ttest_ind.call_count == mock_sample_data.shape[1]


def test_perform_limma_test(mock_sample_data, mock_group1_indices, mock_group2_indices):
    p_values = perform_limma_test(mock_sample_data, mock_group1_indices, mock_group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


def test_perform_limma_test_with_mock(mock_r_functions, mock_sample_data, mock_group1_indices, mock_group2_indices):
    _, _, mock_r = mock_r_functions
    p_values = perform_limma_test(mock_sample_data, mock_group1_indices, mock_group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert mock_r.call_count == 10


def test_perform_rank_sum_test(mock_sample_data, mock_group1_indices, mock_group2_indices):
    p_values = perform_rank_sum_test(mock_sample_data, mock_group1_indices, mock_group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


def test_perform_rank_sum_test_with_mock(mock_ranksums, mock_sample_data, mock_group1_indices, mock_group2_indices):
    p_values = perform_rank_sum_test(mock_sample_data, mock_group1_indices, mock_group2_indices)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert (p_values == 1.0).all()
    assert mock_ranksums.call_count == mock_sample_data.shape[1]


def test_perform_ks_test(mock_sample_data):
    p_values = perform_ks_test(mock_sample_data)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


def test_perform_ks_test_with_mock(mock_kstest, mock_sample_data):
    p_values = perform_ks_test(mock_sample_data)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert (p_values == 1.0).all()
    assert mock_kstest.call_count == mock_sample_data.shape[1]


@pytest.mark.parametrize("test_type", ['t-test', 'limma', 'rank-sum', 'ks-test'])
def test_get_p_values_valid(test_type, mock_sample_data, mock_group1_indices, mock_group2_indices):
    p_values = get_p_values(mock_sample_data, mock_group1_indices, mock_group2_indices, test_type)

    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert ((p_values >= 0) & (p_values <= 1)).all()


@pytest.mark.parametrize("test_type, expected_results", [
    ('t-test', np.array([0.5, 0.5, 0.5])),
    ('limma', np.array([0.4, 0.4, 0.4])),
    ('rank-sum', np.array([0.3, 0.3, 0.3])),
    ('ks-test', np.array([0.2, 0.2, 0.2])),
])
def test_get_p_values_valid_with_mock(test_type, expected_results, mock_statistical_tests, mock_sample_data,
                                      mock_group1_indices,
                                      mock_group2_indices):
    p_values = get_p_values(mock_sample_data, mock_group1_indices, mock_group2_indices, test_type)
    assert mock_statistical_tests[test_type].call_count == 1
    assert isinstance(p_values, np.ndarray)
    assert len(p_values) == mock_sample_data.shape[1]
    assert np.array_equal(p_values, expected_results)


def test_get_p_values_invalid_type(mock_sample_data, mock_group1_indices, mock_group2_indices):
    with pytest.raises(ValueError):
        get_p_values(mock_sample_data, mock_group1_indices, mock_group2_indices, 'invalid-test')


@pytest.mark.parametrize("correction_method", ['bonferroni', 'bh'])
def test_adjust_p_values_valid(mock_p_values, correction_method):
    adjusted_p_values = adjust_p_values(mock_p_values, correction_method)
    assert isinstance(adjusted_p_values, np.ndarray)
    assert len(adjusted_p_values) == mock_p_values.shape[0]
    assert ((adjusted_p_values >= 0) & (adjusted_p_values <= 1)).all()


@pytest.mark.parametrize("correction_method, expected_results", [
    ('bonferroni', np.array([0.5, 0.5, 0.5])),
    ('bh', np.array([0.5, 0.5, 0.5]))
])
def test_adjust_p_values_valid_with_mock(mock_p_values, mock_multipletests, correction_method, expected_results):
    adjusted_p_values = adjust_p_values(mock_p_values, correction_method)
    assert mock_multipletests.call_count == 1
    assert isinstance(adjusted_p_values, np.ndarray)
    assert len(adjusted_p_values) == mock_p_values.shape[0]
    assert np.array_equal(adjusted_p_values, expected_results)


def test_adjust_p_values_invalid_method(mock_p_values):
    with pytest.raises(ValueError):
        adjust_p_values(mock_p_values, 'invalid_method')


@pytest.mark.parametrize("alpha, expected_count", [
    (0.05, 0),
    (0.3, 1),
    (0.6, 2),
    (1.0, 3),
])
def test_count_significant_p_values(mock_adjusted_p_values, alpha, expected_count):
    actual_count = count_significant_p_values(mock_adjusted_p_values, alpha)
    assert actual_count == expected_count


def test_quantify_significance(mock_sample_data, mock_group1_indices, mock_group2_indices, mock_test_type,
                                         mock_correction_method, mock_alpha):
    significant_findings_number = quantify_significance(mock_sample_data, mock_group1_indices, mock_group2_indices,
                                                        mock_test_type,
                                                        mock_correction_method, mock_alpha)
    assert significant_findings_number <= mock_sample_data.shape[1]


def test_quantify_significance_with_mock(mock_sample_data, mock_group1_indices, mock_group2_indices, mock_test_type,
                                         mock_correction_method, mock_alpha, mock_get_p_values, mock_adjust_p_values,
                                         mock_count_significant_p_values):
    significant_findings_number = quantify_significance(mock_sample_data, mock_group1_indices, mock_group2_indices,
                                                        mock_test_type,
                                                        mock_correction_method, mock_alpha)
    assert mock_get_p_values.call_count == 1
    assert mock_adjust_p_values.call_count == 1
    assert mock_count_significant_p_values.call_count == 1
    assert significant_findings_number == 3


if __name__ == '__main__':
    pytest.main([__file__])
