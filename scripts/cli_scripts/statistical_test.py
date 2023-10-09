import argparse
import os

from fdr_hacking.statistical_testing import *
import numpy as np
import json
import pandas as pd


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in YAML format', required=True)
    parser.add_argument('--data_path', help='Path to the input data file', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.replace("\'", "\"")) #TODO: fix this
    simulation_fields = ["n_observations", "n_sites", "dependencies", "correlation_strength", "bin_size_ratio", "id"]
    f_strings = [f'{item}~{config[item]}' for item in simulation_fields]
    f_strings[-1] = f_strings[-1] + ".tsv"
    dataset = os.path.join(args.data_path, *f_strings)
    data = np.loadtxt(dataset, delimiter="\t")
    n_obs = data.shape[0]
    group_size = n_obs // 2
    num_significant_findings = quantify_fdr(methyl_datamat=data,
                                            group1_indices = list(range(group_size)),
                                            group2_indices = list(range(group_size, n_obs)),
                                            test_type=config['statistical_test'],
                                            method=config['multipletest_correction_type'],
                                            alpha=config['alpha'])
    config['num_significant_findings'] = num_significant_findings
    config_df = pd.DataFrame(config, index=[0])
    config_df.to_csv(args.output, sep="\t", index=False)
