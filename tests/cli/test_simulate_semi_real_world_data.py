import ast
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.cli.simulate_semi_real_world_data import parse_args, main, execute

MODULE_PATH = 'scripts.cli.simulate_semi_real_world_data'


def mock_realworld_data():
    return pd.DataFrame({'A': [0.1, 0.2, 0.3, 0.4], 'B': [0.5, 0.6, 0.7, 0.8], 'C': [0.9, 0.10, 0.11, 0.12]})


def mock_sampled_realworld_data():
    return mock_realworld_data().iloc[:, [0, 1]]


def mock_m_values():
    return np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])


@pytest.fixture
def mock_sample_realworld_methylation_values(mocker):
    sampled_df = mock_sampled_realworld_data()
    return mocker.patch(f'{MODULE_PATH}.sample_realworld_methylation_values', return_value=sampled_df)


@pytest.fixture
def mock_load_realworld_data(mocker):
    realworld_data = mock_realworld_data()
    return mocker.patch(f'{MODULE_PATH}.load_realworld_data', return_value=realworld_data)


@pytest.fixture
def mock_beta_to_m(mocker):
    m_values = mock_m_values()
    return mocker.patch(f'{MODULE_PATH}.beta_to_m', return_value=m_values)


@pytest.fixture
def mock_parse_args(mocker):
    mock_args = mocker.MagicMock()
    mock_args.config = '{config}'
    mock_args.output = 'path/to/output.tsv'
    mock_args.config_file_path = 'path/to/config_file_path.yaml'
    mock_args.realworld_data_path = 'path/to/realworld_data_path.h5'
    return mocker.patch(f'{MODULE_PATH}.parse_args', return_value=mock_args)


@pytest.fixture
def mock_main(mocker):
    return mocker.patch(f'{MODULE_PATH}.main')


def test_parse_args(monkeypatch):
    test_args = ['--config', '{config}',
                 '--output', 'path/to/output.tsv',
                 '--config_file_path', 'path/to/config_file_path.yaml',
                 '--realworld_data_path', 'path/to/realworld_data_path.h5']
    monkeypatch.setattr('sys.argv', ['script_name'] + test_args)

    args = parse_args()

    assert args.config == '{config}'
    assert args.output == 'path/to/output.tsv'
    assert args.config_file_path == 'path/to/config_file_path.yaml'
    assert args.realworld_data_path == 'path/to/realworld_data_path.h5'


def test_main_function(mock_load_realworld_data, mock_sample_realworld_methylation_values, mock_beta_to_m, tmp_path):
    config = "{'n_sites': 100}"
    output = tmp_path / "output.tsv"
    config_file_path = tmp_path / "config.yaml"
    realworld_data_path = "path/to/realworld_data.h5"

    main(config, output, config_file_path, realworld_data_path)

    assert os.path.exists(output)
    saved_output = np.loadtxt(output, delimiter="\t")
    assert saved_output.shape == mock_m_values().shape
    assert os.path.exists(config_file_path)
    with open(config_file_path) as f:
        saved_config = yaml.safe_load(f)
    assert saved_config == ast.literal_eval(config)


def test_execute(mock_parse_args, mock_main):
    execute()

    mock_parse_args.called_once()
    mock_main.assert_called_once_with('{config}', 'path/to/output.tsv',
                                      'path/to/config_file_path.yaml', 'path/to/realworld_data_path.h5')


if __name__ == '__main__':
    pytest.main([__file__])
