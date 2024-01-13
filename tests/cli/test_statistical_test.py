import numpy as np
import pytest

from scripts.cli.statistical_test import parse_args, main, execute

MODULE_PATH = 'scripts.cli.statistical_test'


@pytest.fixture
def mock_parse_args(mocker):
    mock_args = mocker.MagicMock()
    mock_args.config = 'path/to/config.yaml'
    mock_args.data_path = 'path/to/data.tsv'
    mock_args.sim_config = 'path/to/sim/config.yaml'
    mock_args.output = 'path/to/output.tsv'
    return mocker.patch(f'{MODULE_PATH}.parse_args', return_value=mock_args)


@pytest.fixture
def mock_config():
    return {
        'statistical_testing': {
            'beta': {
                'statistical_test': ['test1', 'test2'],
                'multipletest_correction_type': ['correction1', 'correction2'],
                'alpha': [0.05, 0.01]
            },
            'normal': {
                'statistical_test': ['test1', 'test2'],
                'multipletest_correction_type': ['correction1', 'correction2'],
                'alpha': [0.05, 0.01]
            }
        }
    }


@pytest.fixture
def mock_load_txt(mocker):
    return mocker.patch('numpy.loadtxt', return_value=np.random.rand(100, 4))


@pytest.fixture
def mock_csv_write(mocker):
    return mocker.patch('pandas.DataFrame.to_csv')


@pytest.fixture
def mock_quantify_significance(mocker):
    return mocker.patch(f'{MODULE_PATH}.quantify_significance', return_value=0)


@pytest.fixture
def mock_main(mocker):
    return mocker.patch(f'{MODULE_PATH}.main')

def test_parse_args(monkeypatch):
    test_args = ['--config', 'path/to/config.yaml',
                 '--data_path', 'path/to/data.tsv',
                 '--sim_config', 'path/to/sim/config.yaml',
                 '--output', 'path/to/output.tsv']
    monkeypatch.setattr('sys.argv', ['script_name'] + test_args)

    args = parse_args()

    assert args.config == 'path/to/config.yaml'
    assert args.data_path == 'path/to/data.tsv'
    assert args.sim_config == 'path/to/sim/config.yaml'
    assert args.output == 'path/to/output.tsv'


@pytest.mark.parametrize("sim_config", [
    {'data_distribution': 'beta', 'multipletest_correction_type': 'bh', 'alpha': 0.05},
    {'data_distribution': 'normal', 'multipletest_correction_type': 'bh', 'alpha': 0.05}
])
def test_main(mocker, sim_config, mock_config, mock_load_txt, mock_csv_write, mock_quantify_significance):
    mock_parse_yaml_file = mocker.patch(f'{MODULE_PATH}.parse_yaml_file', side_effect=[mock_config, sim_config])
    main('config_path', 'data_path', 'sim_config_path', 'output_path')

    assert mock_parse_yaml_file.call_count == 2
    mock_parse_yaml_file_first_call_args, _ = mock_parse_yaml_file.call_args_list[0]
    assert mock_parse_yaml_file_first_call_args == ('config_path',)
    mock_parse_yaml_file_second_call_args, _ = mock_parse_yaml_file.call_args_list[1]
    assert mock_parse_yaml_file_second_call_args == ('sim_config_path',)
    mock_load_txt.assert_called_once_with('data_path', delimiter="\t")
    assert mock_quantify_significance.call_count == 8
    mock_csv_write.assert_called_once_with('output_path', sep="\t", index=False)


def test_execute(mock_parse_args, mock_main):
    execute()

    mock_parse_args.called_once()
    mock_main.assert_called_once_with('path/to/config.yaml', 'path/to/data.tsv', 'path/to/sim/config.yaml', 'path/to/output.tsv')


if __name__ == '__main__':
    pytest.main([__file__])
