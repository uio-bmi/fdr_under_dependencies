import argparse
from itertools import product

import pandas as pd

from scripts.analysis.statistical_testing import *
from scripts.analysis.utils import parse_yaml_file


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
    n_obs = data.shape[0]
    group_size = n_obs // 2
    statistical_test_params = product(config['statistical_test'], config['multipletest_correction_type'],
                                      config['alpha'])
    fdr_full_results = []
    for params in statistical_test_params:
        fdr_results = sim_config.copy()
        fdr_results['statistical_test'], fdr_results['multipletest_correction_type'], fdr_results['alpha'] = params
        num_significant_findings = quantify_fdr(methyl_datamat=data,
                                                group1_indices=list(range(group_size)),
                                                group2_indices=list(range(group_size, n_obs)),
                                                test_type=fdr_results['statistical_test'],
                                                method=fdr_results['multipletest_correction_type'],
                                                alpha=fdr_results['alpha'])
        fdr_results['num_significant_findings'] = num_significant_findings
        fdr_full_results.append(fdr_results)

    fdr_results_df = pd.DataFrame(fdr_full_results)
    fdr_results_df.to_csv(args.output, sep="\t", index=False)
