import argparse
from itertools import product

import numpy as np
import pandas as pd

from scripts.analysis.statistical_analysis import quantify_significance
from scripts.analysis.utils import parse_yaml_file


def parse_args():
    """
    This function parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in YAML format', required=True)
    parser.add_argument('--data_path', help='Path to the input data file in tsv format', required=True)
    parser.add_argument('--sim_config', help='Path to the simulation config file in yaml format', required=True)
    parser.add_argument('--output', help='Path to the output file in tsv format', required=True)
    args = parser.parse_args()
    return args


def main(config, data_path, sim_config, output):
    """
    This function performs statistical testing with FDR/FWR correction and saves the results to a file.

    :param config: Path to the configuration file in yaml format
    :param data_path: Path to the input data file in tsv format
    :param sim_config: Path to the simulation config file in yaml format
    :param output: Path to the output file in tsv format
    :return: None
    """
    config = parse_yaml_file(config)
    sim_config = parse_yaml_file(sim_config)
    data = np.loadtxt(data_path, delimiter="\t")
    if sim_config['data_distribution'] == 'beta':
        statistical_test_config = config['statistical_testing']['beta']
    else:
        statistical_test_config = config['statistical_testing']['normal']
    n_observations = data.shape[0]
    group_size = n_observations // 2
    statistical_test_params = product(statistical_test_config['statistical_test'],
                                      statistical_test_config['multipletest_correction_type'],
                                      statistical_test_config['alpha'])
    fdr_full_results = []
    for params in statistical_test_params:
        fdr_results = sim_config.copy()
        fdr_results['statistical_test'], fdr_results['multipletest_correction_type'], fdr_results['alpha'] = params
        num_significant_findings = quantify_significance(data=data,
                                                         group1_indices=list(range(group_size)),
                                                         group2_indices=list(range(group_size, n_observations)),
                                                         test_type=fdr_results['statistical_test'],
                                                         method=fdr_results['multipletest_correction_type'],
                                                         alpha=fdr_results['alpha'])
        fdr_results['num_significant_findings'] = num_significant_findings
        fdr_full_results.append(fdr_results)

    fdr_results_df = pd.DataFrame(fdr_full_results)
    fdr_results_df.to_csv(output, sep="\t", index=False)


def execute():
    """
    This function is executed when this file is run as a script.
    """
    args = parse_args()
    main(args.config, args.data_path, args.sim_config, args.output)
