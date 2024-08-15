import pandas as pd
import pytest

from scripts.cli.plot_histograms import parse_args, main, execute


def mock_aggregated_results():
    df = pd.DataFrame({
        'n_observations': [100],
        'n_sites': [10],
        'reporting_histogram_bins': ['[0, 1, 5, 10]'],
        'reporting_histogram': ['[0, 1, 1]'],
        'config_id': ['config_id']
    })
    return df


@pytest.fixture
def mock_read_csv(mocker):
    return mocker.patch('pandas.read_csv', return_value=mock_aggregated_results())


@pytest.fixture
def mock_os_mkdir(mocker):
    return mocker.patch('os.mkdir')


@pytest.fixture
def mock_os_exists(mocker):
    return mocker.patch('os.path.exists', return_value=False)


@pytest.fixture
def mock_go_bar(mocker):
    return mocker.patch('plotly.graph_objs.Bar')


@pytest.fixture
def mock_go_layout(mocker):
    return mocker.patch('plotly.graph_objs.Layout')


@pytest.fixture
def mock_go_figure(mocker):
    figure_mock = mocker.MagicMock()
    figure_mock.write_image = mocker.MagicMock()
    return mocker.patch('plotly.graph_objs.Figure', return_value=figure_mock)


@pytest.fixture
def mock_parse_args(mocker):
    mock_args = mocker.MagicMock()
    mock_args.aggregated_results = 'path/to/aggregated_results.tsv'
    mock_args.output_dir = 'path/to/output_dir'
    mock_args.with_title = True
    mock_args.remove_zero_bin = True
    return mocker.patch('scripts.cli.plot_histograms.parse_args', return_value=mock_args)


@pytest.fixture
def mock_main(mocker):
    return mocker.patch('scripts.cli.plot_histograms.main')


def test_parse_args(monkeypatch):
    test_args = [
        '--aggregated_results', 'path/to/aggregated_results.tsv',
        '--output_dir', 'path/to/output_dir',
        '--with_title',
        '--remove_zero_bin'
    ]

    monkeypatch.setattr('sys.argv', ['script_name'] + test_args)

    args = parse_args()

    assert args.aggregated_results == 'path/to/aggregated_results.tsv'
    assert args.output_dir == 'path/to/output_dir'
    assert args.with_title is True
    assert args.remove_zero_bin is True


@pytest.mark.parametrize("with_title, remove_zero_bin", [(True, True), (False, False)])
def test_main(mock_read_csv, mock_os_mkdir, mock_os_exists, mock_go_bar, mock_go_layout, mock_go_figure,
              with_title, remove_zero_bin):
    aggregated_file = 'path/to/aggregated_results.tsv'
    output_dir = 'path/to/output_dir'

    main(aggregated_file, output_dir, with_title, remove_zero_bin)

    mock_read_csv.assert_called_once_with(aggregated_file, sep="\t", header=0, index_col=False)
    mock_os_mkdir.assert_called_once_with(output_dir)
    mock_os_exists.assert_called_once_with(output_dir)
    mock_go_bar.assert_called_once()
    mock_go_layout.assert_called_once()
    mock_go_figure.assert_called_once()


def test_execute(mock_parse_args, mock_main):
    execute()

    mock_parse_args.assert_called_once()
    mock_main.assert_called_once_with('path/to/aggregated_results.tsv', 'path/to/output_dir', True, True)


if __name__ == '__main__':
    pytest.main([__file__])
