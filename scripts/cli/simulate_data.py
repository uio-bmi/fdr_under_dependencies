import argparse
import ast

import numpy as np
import yaml

from scripts.analysis.data_generation import load_realworld_data, simulate_methyl_data, \
    synthesize_gaussian_dataset_without_dependence, synthesize_correlated_gaussian_bins


def parse_args():
    """
    This function parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in string dictionary like format', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    parser.add_argument('--config_file_path', help='Path to the simulation config file', required=True)
    parser.add_argument('--realworld_data_path', help='Path to the real-world dataset in H5 file format',
                        required=True)
    args = parser.parse_args()
    return args


def main(config, output, config_file_path, realworld_data_path):
    """
    This function generates synthetic data and saves it to a file.

    :param config: Path to the configuration file in yaml format
    :param output: Path to the output file
    :param config_file_path: Path to the yaml file where specific parameters for generating data should be saved
    :param realworld_data_path: Path to the real-world dataset in H5 file format
    :return: None
    """
    config = ast.literal_eval(config)
    realworld_data = load_realworld_data(file_path=realworld_data_path)
    if config['dependencies'] == 0:
        if config['data_distribution'] == 'beta':
            simulated_data = simulate_methyl_data(realworld_data, config['n_sites'], config['n_observations'],
                                                  config['dependencies'],
                                                  None, None)
        else:
            simulated_data = synthesize_gaussian_dataset_without_dependence(n_observations=config['n_observations'],
                                                                            n_sites=config['n_sites'])
    else:
        if not isinstance(config['bin_size_ratio'], float):
            raise ValueError("bin_size_ratio must be a float")
        if not config['correlation_strength'] in ['medium', 'high']:
            raise ValueError("Dependencies must be medium or high")

        correlation_coefficient_distribution = []
        if config['correlation_strength'] == 'medium':
            correlation_coefficient_distribution = [(-0.6, -0.4), (-0.1, 0.1), (0.4, 0.6)]
        elif config['correlation_strength'] == 'high':
            correlation_coefficient_distribution = [(-0.85, -0.7), (-0.1, 0.1), (0.7, 0.85)]

        if config['data_distribution'] == 'beta':
            simulated_data = simulate_methyl_data(realworld_data, config['n_sites'], config['n_observations'],
                                                  config['dependencies'],
                                                  int(config['bin_size_ratio'] * config['n_sites']),
                                                  correlation_coefficient_distribution)
        else:
            simulated_data = synthesize_correlated_gaussian_bins(correlation_coefficient_distribution=
                                                                 correlation_coefficient_distribution,
                                                                 n_observations=config['n_observations'],
                                                                 n_sites=config['n_sites'],
                                                                 bin_size=int(
                                                                     config['bin_size_ratio'] * config['n_sites']))
    np.savetxt(output, simulated_data, delimiter="\t")
    with open(config_file_path, "w+") as yaml_file:
        yaml.dump(config, yaml_file)


def execute():
    """
    This function is executed when this file is run as a script.
    """
    args = parse_args()
    main(args.config, args.output, args.config_file_path, args.realworld_data_path)
