import numpy as np
import pytest

from scripts.cli.execute_statistical_test import execute, parse_args, execute_statistical_test_on_single_dataset, main

MODULE_PATH = 'scripts.cli.execute_statistical_test'


def mock_dataset():
    return np.array([[1, 2], [3, 4], [5, 6], [7, 8]])


@pytest.fixture
def mock_loadtxt(mocker):
    return mocker.patch('numpy.loadtxt', return_value=mock_dataset())


@pytest.fixture
def mock_savetxt(mocker):
    return mocker.patch('numpy.savetxt')


@pytest.fixture
def mock_ttest_ind(mocker):
    return mocker.patch(f'{MODULE_PATH}.ttest_ind', return_value=(1.0, 1.0))


@pytest.fixture
def mock_plotly_express_histogram(mocker):
    plot_mock = mocker.MagicMock()
    plot_mock.write_image = mocker.MagicMock()

    return mocker.patch('plotly.express.histogram', return_value=plot_mock)


@pytest.fixture
def mock_plotly_ecdf(mocker):
    plot_mock = mocker.MagicMock()
    plot_mock.write_image = mocker.MagicMock()

    return mocker.patch('plotly.express.ecdf', return_value=plot_mock)


@pytest.fixture
def mock_execute_statistical_test_on_single_dataset(mocker):
    return mocker.patch(
        f'{MODULE_PATH}.execute_statistical_test_on_single_dataset',
        return_value=(np.array([0.1, 0.2, 0.3]), np.array([1, 2, 3]))
    )


@pytest.fixture
def mock_glob(mocker):
    return mocker.patch('glob.glob', return_value=['dataset1.tsv', 'dataset2.tsv'])


@pytest.fixture
def mock_csv_write(mocker):
    return mocker.patch('pandas.DataFrame.to_csv')


@pytest.fixture
def mock_parse_args(mocker):
    mock_args = mocker.MagicMock()
    mock_args.data_path = 'path/to/data/dir'
    mock_args.output = 'path/to/output/dir'
    return mocker.patch(f'{MODULE_PATH}.parse_args', return_value=mock_args)


@pytest.fixture
def mock_main(mocker):
    return mocker.patch(f'{MODULE_PATH}.main')


def test_parse_args(monkeypatch):
    test_args = ['--data_path', 'path/to/data/dir',
                 '--output', 'path/to/output/dir']
    monkeypatch.setattr('sys.argv', ['script_name'] + test_args)

    args = parse_args()

    assert args.data_path == 'path/to/data/dir'
    assert args.output == 'path/to/output/dir'


def test_execute_statistical_test_on_single_dataset(mock_loadtxt, mock_savetxt, mock_ttest_ind,
                                                    mock_plotly_express_histogram):

    dataset = mock_dataset()

    p_values, test_statistics = execute_statistical_test_on_single_dataset("dummy_path", "dummy_output")

    assert len(p_values) == dataset.shape[1]
    assert len(test_statistics) == dataset.shape[1]
    assert mock_loadtxt.called_once()
    assert mock_savetxt.call_count == 2
    assert mock_ttest_ind.call_count == dataset.shape[1]
    assert mock_plotly_express_histogram.call_count == 2


def test_main(mock_glob, mock_execute_statistical_test_on_single_dataset, mock_plotly_ecdf, mock_csv_write):
    main('path/to/data/dir', 'path/to/output/dir')

    assert mock_glob.called_once()
    assert mock_execute_statistical_test_on_single_dataset.call_count == len(mock_glob.return_value)
    assert mock_plotly_ecdf.called_once()
    assert mock_csv_write.call_count == 2


def test_execute(mock_parse_args, mock_main):
    execute()

    mock_parse_args.called_once()
    mock_main.called_once_with('path/to/data/dir', 'path/to/output/dir')


if __name__ == '__main__':
    pytest.main([__file__])
