from itertools import product

import pandas as pd


def append_params(config_dict: dict):
    """
    Given a dictionary of the form {param_name: [param_value1, param_value2, ...]}, this function returns a dataframe
    with all possible combinations of the parameters

    :param config_dict: A dictionary of the form {param_name: [param_value1, param_value2, ...]}
    :return: A dataframe with all possible combinations of the parameters
    """
    if config_dict.get('n_datasets') is None:
        raise ValueError("The config_dict must contain the key 'n_datasets'")
    n_datasets = config_dict['n_datasets']
    del config_dict['n_datasets']
    workflow_unique_params = list(product(*config_dict.values()))
    ids = [i for i in range(1, n_datasets + 1) for _ in range(len(workflow_unique_params))]
    params_df = pd.DataFrame(workflow_unique_params * n_datasets, columns = config_dict.keys())
    params_df['id'] = ids
    return params_df


def generate_simulation_params(workflow_config: dict, workflow_params_file: str):
    """
    Given a dictionary of the form {param_name: [param_value1, param_value2, ...]}, this function generates a dataframe
    with all possible combinations of the parameters and saves it to a csv file.

    :param workflow_config: A dictionary of the form {param_name: [param_value1, param_value2, ...]}
    :param workflow_params_file: The path to the tsv file where the dataframe will be saved
    :return: None
    """
    list_of_dfs = []
    for value in workflow_config.values():
        list_of_dfs.append(append_params(value))
    params_df = pd.concat(list_of_dfs, axis=0)
    params_df.to_csv(workflow_params_file, sep="\t", index=False)
