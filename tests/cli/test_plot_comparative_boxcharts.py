import pandas as pd
import pytest

from scripts.cli.plot_comparative_boxcharts import parse_args, without, execute, main

MODULE_PATH = 'scripts.cli.plot_comparative_boxcharts'


@pytest.fixture
def mock_parse_args(mocker):
    mock_args = mocker.MagicMock()
    mock_args.concatenated_results = 'path/to/concatenated_results.tsv'
    mock_args.reporting_config_file = 'path/to/reporting_config_file.yaml'
    mock_args.output_dir = 'path/to/output_dir'
    return mocker.patch(f'{MODULE_PATH}.parse_args', return_value=mock_args)


@pytest.fixture
def mock_main(mocker):
    return mocker.patch(f'{MODULE_PATH}.main')


def mock_concatenated_results():
    df = pd.DataFrame({
        'alpha': [0.05, 0.05],
        'multipletest_correction_type': ['fdr_bh', 'fdr_bh'],
        'statistical_test': ['pearson', 'pearson'],
        'bin_size_ratio': [0.1, 0.1],
        'correlation_strength': [0.5, 0.5],
        'n_observations': [100, 100],
        'n_sites': [10, 10],
        'id': [0, 1],
        'num_significant_findings': [3, 7],
        'reporting_histogram_bins': ['[0, 1, 5, 10]', '[0, 1, 5, 10]']
    })
    return df


def mock_reporting_config():
    return {
        'reporting':
            {'compare_bin_size_ratio':
                {
                    'alpha': [0.05],
                    'multipletest_correction_type': ['fdr_bh'],
                    'statistical_test': ['pearson'],
                    'bin_size_ratio': [0.1, 0.2],
                    'correlation_strength': [0.5, 0.6],
                    'n_sites': [10],
                    'n_observations': [100],
                    'x_axis': 'bin_size_ratio'
                }
            }
    }


def mock_reporting_config_without_x_axis():
    return {
        'alpha': [0.05],
        'multipletest_correction_type': ['fdr_bh'],
        'statistical_test': ['pearson'],
        'bin_size_ratio': [0.1, 0.2],
        'correlation_strength': [0.5, 0.6],
        'n_sites': [10],
        'n_observations': [100]
    }


@pytest.fixture
def mock_without(mocker):
    return mocker.patch(f'{MODULE_PATH}.without', return_value=mock_reporting_config_without_x_axis())


@pytest.fixture
def mock_parse_yaml_file(mocker):
    return mocker.patch(f'{MODULE_PATH}.parse_yaml_file', return_value=mock_reporting_config())


@pytest.fixture
def mock_without_with_dependencies(mocker):
    return_dict = mock_reporting_config_without_x_axis()
    return_dict['dependencies'] = [True]
    return mocker.patch(f'{MODULE_PATH}.without', return_value=return_dict)


@pytest.fixture
def mock_parse_yaml_file_with_dependencies(mocker):
    return_dict = mock_reporting_config()
    return_dict['reporting']['compare_bin_size_ratio']['dependencies'] = [True]
    return mocker.patch(f'{MODULE_PATH}.parse_yaml_file', return_value=return_dict)


@pytest.fixture
def mock_csv_read(mocker):
    return mocker.patch('pandas.read_csv', return_value=mock_concatenated_results())


@pytest.fixture
def mock_csv_read_with_dependencies(mocker):
    return_df = mock_concatenated_results()
    return_df['dependencies'] = [True, True]
    return mocker.patch('pandas.read_csv', return_value=return_df)


@pytest.fixture
def mock_os_exists(mocker):
    return mocker.patch('os.path.exists', return_value=False)


@pytest.fixture
def mock_os_mkdir(mocker):
    return mocker.patch('os.mkdir')


@pytest.fixture
def mock_px_box(mocker):
    plot_mock = mocker.MagicMock()
    plot_mock.write_image = mocker.MagicMock()
    plot_mock.write_html = mocker.MagicMock()

    return mocker.patch('plotly.express.box', return_value=plot_mock)


def test_parse_args(monkeypatch):
    test_args = ['--concatenated_results', 'path/to/concatenated_results.tsv',
                 '--reporting_config_file', 'path/to/reporting_config_file.yaml',
                 '--output_dir', 'path/to/output_dir']

    monkeypatch.setattr('sys.argv', ['script_name'] + test_args)

    args = parse_args()

    assert args.concatenated_results == 'path/to/concatenated_results.tsv'
    assert args.reporting_config_file == 'path/to/reporting_config_file.yaml'
    assert args.output_dir == 'path/to/output_dir'


def test_without():
    test_dict = {'a': 1, 'b': 2, 'c': 3}
    test_key = 'b'

    test_dict_without_key = without(test_dict, test_key)

    assert test_dict_without_key == {'a': 1, 'c': 3}


def test_main_without_dependencies(mock_csv_read, mock_parse_yaml_file, mock_without,
                                   mock_os_exists, mock_os_mkdir, mock_px_box):
    main('path/to/concatenated_results.tsv', 'path/to/reporting_config_file.yaml', 'path/to/output_dir')

    assert mock_csv_read.call_count == 1
    assert mock_parse_yaml_file.call_count == 1
    assert mock_os_exists.call_count == 1
    assert mock_os_mkdir.call_count == 1
    assert mock_without.call_count == 1
    assert mock_px_box.call_count == 1


def test_main_with_dependencies(mock_csv_read, mock_csv_read_with_dependencies, mock_parse_yaml_file,
                                mock_parse_yaml_file_with_dependencies, mock_without, mock_without_with_dependencies,
                                mock_os_exists, mock_os_mkdir, mock_px_box):
    main('path/to/concatenated_results.tsv', 'path/to/reporting_config_file.yaml', 'path/to/output_dir')

    assert mock_csv_read_with_dependencies.call_count == 1
    assert mock_parse_yaml_file_with_dependencies.call_count == 1
    assert mock_os_exists.call_count == 1
    assert mock_os_mkdir.call_count == 1
    assert mock_without_with_dependencies.call_count == 1
    assert mock_px_box.call_count == 1


def test_execute(mock_parse_args, mock_main):
    execute()

    assert mock_parse_args.call_count == 1
    assert mock_main.called_once_with('path/to/concatenated_results.tsv', 'path/to/reporting_config_file.yaml',
                                      'path/to/output_dir')


if __name__ == '__main__':
    pytest.main([__file__])
