import ast
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.cli.simulate_data import execute, parse_args, main

MODULE_PATH = 'scripts.cli.simulate_data'


def mock_realworld_data():
    return pd.DataFrame({'A': [0.1, 0.2, 0.3, 0.4], 'B': [0.5, 0.6, 0.7, 0.8], 'C': [0.9, 0.10, 0.11, 0.12]})


def mock_simulated_data():
    return np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])


@pytest.fixture
def mock_simulate_methyl_data(mocker):
    return mocker.patch(f'{MODULE_PATH}.simulate_methyl_data', return_value=mock_simulated_data())


@pytest.fixture
def mock_synthesize_gaussian_dataset_without_dependence(mocker):
    return mocker.patch(f'{MODULE_PATH}.synthesize_gaussian_dataset_without_dependence',
                        return_value=mock_simulated_data())


@pytest.fixture
def mock_synthesize_correlated_gaussian_bins(mocker):
    return mocker.patch(f'{MODULE_PATH}.synthesize_correlated_gaussian_bins', return_value=mock_simulated_data())


@pytest.fixture
def mock_load_realworld_data(mocker):
    realworld_data = mock_realworld_data()
    return mocker.patch(f'{MODULE_PATH}.load_realworld_data', return_value=realworld_data)


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


@pytest.mark.parametrize("config, mock_to_assert", [
    ("{'dependencies': 0, 'data_distribution': 'beta', 'n_sites': 100, 'n_observations': 200, "
     "'bin_size_ratio': 0.1, 'correlation_strength': 'medium'}",
     'mock_simulate_methyl_data'),
    ("{'dependencies': 0, 'data_distribution': 'gaussian', 'n_sites': 100, 'n_observations': 200, "
     "'bin_size_ratio': 0.1, 'correlation_strength': 'medium'}",
     'mock_synthesize_gaussian_dataset_without_dependence'),
    ("{'dependencies': 1, 'data_distribution': 'beta', 'n_sites': 100, 'n_observations': 200, "
     "'bin_size_ratio': 0.1, 'correlation_strength': 'medium'}",
     'mock_simulate_methyl_data'),
    ("{'dependencies': 1, 'data_distribution': 'gaussian', 'n_sites': 100, 'n_observations': 200, "
     "'bin_size_ratio': 0.1, 'correlation_strength': 'high'}",
     'mock_synthesize_correlated_gaussian_bins')
])
def test_main_various_configs(mock_load_realworld_data, mock_simulate_methyl_data,
                              mock_synthesize_gaussian_dataset_without_dependence,
                              mock_synthesize_correlated_gaussian_bins, tmp_path, config, mock_to_assert):
    output = tmp_path / "output.tsv"
    config_file_path = tmp_path / "config.yaml"
    realworld_data_path = "dummy_realworld_data_path"

    main(config, output, config_file_path, realworld_data_path)

    assert locals()[mock_to_assert].call_count == 1
    assert os.path.exists(output)
    saved_output = np.loadtxt(output, delimiter="\t")
    assert saved_output.shape == mock_simulated_data().shape
    assert os.path.exists(config_file_path)
    with open(config_file_path) as f:
        saved_config = yaml.safe_load(f)
    assert saved_config == ast.literal_eval(config)


def test_main_invalid_bin_size_ratio(mock_load_realworld_data):
    config = ("{'dependencies':1, 'bin_size_ratio': 'not_a_float', 'correlation_strength': 'medium', 'n_sites': 100, "
              "'n_observations': 100}")
    output = "dummy_output_path"
    config_file_path = "dummy_config_file_path"
    realworld_data_path = "dummy_realworld_data_path"

    with pytest.raises(ValueError) as value_error:
        main(config, output, config_file_path, realworld_data_path)
    assert "bin_size_ratio must be a float" in str(value_error.value)


def test_main_invalid_correlation_strength(mock_load_realworld_data):
    config = ("{'dependencies':1, 'bin_size_ratio': 0.1, 'correlation_strength': 'not_medium_or_high', 'n_sites': 100, "
              "'n_observations': 100}")
    output = "dummy_output_path"
    config_file_path = "dummy_config_file_path"
    realworld_data_path = "dummy_realworld_data_path"

    with pytest.raises(ValueError) as value_error:
        main(config, output, config_file_path, realworld_data_path)
    assert "Dependencies must be medium or high" in str(value_error.value)


def test_execute(mock_parse_args, mock_main):
    execute()

    mock_parse_args.called_once()
    mock_main.assert_called_once_with('{config}', 'path/to/output.tsv',
                                      'path/to/config_file_path.yaml', 'path/to/realworld_data_path.h5')


if __name__ == '__main__':
    pytest.main([__file__])
