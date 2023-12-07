from itertools import product, chain

import pandas as pd


def append_params(config_dict: dict):
    n_datasets = config_dict['n_datasets']
    del config_dict['n_datasets']
    config_values = config_dict.values()
    config_keys = config_dict.keys()
    workflow_unique_params = (list(product(*config_values)))
    workflow_params = workflow_unique_params * n_datasets
    ids = [[i] * len(workflow_unique_params) for i in range(1, n_datasets+1)]
    ids = list(chain(*ids))
    df = pd.DataFrame(workflow_params, columns = config_keys)
    df['id'] = ids
    return df


def generate_simulation_params(workflow_config: dict, workflow_params_file: str):
    # workflow_config = parse_yaml_file(workflow_yaml_file)
    list_of_dfs = []
    for val in workflow_config.values():
        list_of_dfs.append(append_params(val))
    params_df = pd.concat(list_of_dfs, axis=0)
    params_df.to_csv(workflow_params_file, sep="\t", index=False)
