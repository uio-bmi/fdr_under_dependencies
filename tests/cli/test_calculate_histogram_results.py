import pandas as pd
import pytest

from scripts.cli.calculate_histogram_results import parse_args, main, execute

MODULE_PATH = 'scripts.cli.calculate_histogram_results'


def mock_input_df():
    df = pd.DataFrame({
        'n_observations': [100, 100],
        'n_sites': [10, 10],
        'id': [0, 1],
        'num_significant_findings': [3, 7],
        'reporting_histogram_bins': ['[0, 1, 5, 10]', '[0, 1, 5, 10]']
    })
    return df


def mock_input_file(tmp_path):
    df = mock_input_df()
    input_file = tmp_path / "input.tsv"
    df.to_csv(input_file, sep="\t", index=False)
    return input_file


@pytest.fixture
def mock_parse_args(mocker):
    mock_args = mocker.MagicMock()
    mock_args.concatenated_results = 'path/to/concatenated_results.tsv'
    mock_args.aggregated_results = 'path/to/aggregated_results.tsv'
    return mocker.patch(f'{MODULE_PATH}.parse_args', return_value=mock_args)


@pytest.fixture
def mock_main(mocker):
    return mocker.patch(f'{MODULE_PATH}.main')


@pytest.fixture
def mock_csv_read(mocker):
    return mocker.patch('pandas.read_csv', return_value=mock_input_df())


@pytest.fixture
def mock_csv_write(mocker):
    return mocker.patch('pandas.DataFrame.to_csv')


def test_parse_args(monkeypatch):
    test_args = ['--concatenated_results', 'path/to/concatenated_results.tsv',
                 '--aggregated_results', 'path/to/aggregated_results.tsv']
    monkeypatch.setattr('sys.argv', ['script_name'] + test_args)

    args = parse_args()

    assert args.concatenated_results == 'path/to/concatenated_results.tsv'
    assert args.aggregated_results == 'path/to/aggregated_results.tsv'


def test_main(tmp_path):
    input_file = mock_input_file(tmp_path)
    output_file = tmp_path / "output.tsv"

    main(str(input_file), str(output_file))
    output_df = pd.read_csv(output_file, sep="\t")

    assert not output_df.empty
    assert 'config_id' in output_df.columns
    assert 'reporting_histogram' in output_df.columns
    assert 'n_observations~100__n_sites~10' in output_df['config_id'].values
    assert '[0, 1, 1]' in output_df['reporting_histogram'].values


def test_main_with_mock(mock_csv_read, mock_csv_write):
    main('dummy_input_path', 'dummy_output_path')

    mock_csv_read.assert_called_once()
    mock_csv_write.assert_called_once()


def test_execute(mock_parse_args, mock_main):
    execute()

    mock_parse_args.called_once()
    mock_main.assert_called_once_with('path/to/concatenated_results.tsv', 'path/to/aggregated_results.tsv')


if __name__ == '__main__':
    pytest.main([__file__])
