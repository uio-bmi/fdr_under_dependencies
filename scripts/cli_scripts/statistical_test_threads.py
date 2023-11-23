import argparse
import concurrent.futures
import os
from itertools import product
from fdr_hacking.statistical_testing import *
import numpy as np
import pandas as pd
from fdr_hacking.util import parse_yaml_file


def perform_statistical_test(params):
    data, sim_config, test_config = params
    n_obs = data.shape[0]
    group_size = n_obs // 2
    fdr_results = sim_config.copy()
    fdr_results['statistical_test'], fdr_results['multipletest_correction_type'], fdr_results['alpha'] = test_config
    num_significant_findings = quantify_fdr(methyl_datamat=data,
                                            group1_indices=list(range(group_size)),
                                            group2_indices=list(range(group_size, n_obs)),
                                            test_type=fdr_results['statistical_test'],
                                            method=fdr_results['multipletest_correction_type'],
                                            alpha=fdr_results['alpha'])
    fdr_results['num_significant_findings'] = num_significant_findings
    return fdr_results


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in YAML format', required=True)
    parser.add_argument('--data_path', help='Path to the input data file', required=True)
    parser.add_argument('--sim_config', help='Path to the simulation config file', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    args = parser.parse_args()
    config = parse_yaml_file(args.config)
    sim_config = parse_yaml_file(args.sim_config)
    data = np.loadtxt(args.data_path, delimiter="\t")
    if sim_config['data_distribution'] == 'beta':
        config = config['statistical_testing']['beta']
    else:
        config = config['statistical_testing']['normal']

    statistical_test_params = product(config['statistical_test'], config['multipletest_correction_type'],
                                      config['alpha'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        params = [(data, sim_config, param) for param in statistical_test_params]
        fdr_full_results = list(executor.map(perform_statistical_test, params))

    fdr_results_df = pd.DataFrame(fdr_full_results)
    fdr_results_df.to_csv(args.output, sep="\t", index=False)
