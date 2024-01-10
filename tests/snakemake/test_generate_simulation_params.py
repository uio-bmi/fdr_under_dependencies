import os
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from scripts.snakemake.generate_simulation_params import append_params, generate_simulation_params


@pytest.mark.parametrize("config_dict, expected_data, expected_columns", [
    ({
        'n_datasets': 2,
        'param1': [1],
        'param2': ['a', 'b']
    }, [
        [1, 'a', 1],
        [1, 'b', 1],
        [1, 'a', 2],
        [1, 'b', 2],
    ], ['param1', 'param2', 'id']),
    ({
        'n_datasets': 1,
        'param1': [1, 2],
        'param2': ['a']
    }, [
        [1, 'a', 1],
        [2, 'a', 1],
    ], ['param1', 'param2', 'id'])
])
def test_append_params(config_dict, expected_data, expected_columns):
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    result_df = append_params(config_dict)

    pd.testing.assert_frame_equal(result_df, expected_df)


def test_append_params_invalid_config_dict():
    with pytest.raises(ValueError, match="The config_dict must contain the key 'n_datasets'"):
        append_params({})


@pytest.mark.parametrize("workflow_config, expected_row_count, expected_columns", [
    ({
        'workflow_1': {
            'n_datasets': 2,
            'param1': [1],
            'param2': ['a', 'b']
        },
        'workflow_2': {
            'n_datasets': 2,
            'param1': [2, 3],
            'param2': ['c']
        }
    }, 8, ['param1', 'param2', 'id']),
])
def test_generate_simulation_params(workflow_config, expected_row_count, expected_columns):
    with TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test_output.csv")
        generate_simulation_params(workflow_config, output_file)

        assert os.path.exists(output_file)
        result_df = pd.read_csv(output_file, sep="\t")
        assert len(result_df) == expected_row_count
        assert list(result_df.columns) == expected_columns


if __name__ == '__main__':
    pytest.main([__file__])
