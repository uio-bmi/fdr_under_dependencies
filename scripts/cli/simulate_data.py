import argparse
import json

import yaml

from scripts.analysis.data_generation import *


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in YAML format', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    parser.add_argument('--config_file_path', help='Path to the simulation config file', required=True)
    parser.add_argument('--realworld_data_path', help='Path to the real-world dataset in H5 file format',
                        required=True)
    args = parser.parse_args()
    config = json.loads(args.config.replace("\'", "\""))  # TODO: fix this
    realworld_data = load_realworld_data(file_path=args.realworld_data_path)
    if config['dependencies'] == 0:
        if config['data_distribution'] == 'beta':
            simulated_data = simulate_methyl_data(realworld_data, config['n_sites'], config['n_observations'],
                                                  config['dependencies'],
                                                  None, None)
        else:
            simulated_data = synthesize_gaussian_dataset_without_dependence(n_observations=config['n_observations'],
                                                                            n_sites=config['n_sites'])
    else:
        assert isinstance(config['bin_size_ratio'], float), "bin_size_ratio must be a float"
        assert config['correlation_strength'] in ['medium', 'high'], "dependencies must be either medium or high"
        if config['correlation_strength'] == 'medium':
            corr_coef_dist = [(-0.6, -0.4), (-0.1, 0.1), (0.4, 0.6)]
        elif config['correlation_strength'] == 'high':
            corr_coef_dist = [(-0.85, -0.7), (-0.1, 0.1), (0.7, 0.85)]
        if config['data_distribution'] == 'beta':
            simulated_data = simulate_methyl_data(realworld_data, config['n_sites'], config['n_observations'],
                                                  config['dependencies'],
                                                  int(config['bin_size_ratio'] * config['n_sites']), corr_coef_dist)
        else:
            simulated_data = synthesize_correlated_gaussian_bins(corr_coef_distribution=corr_coef_dist,
                                                                 n_observations=config['n_observations'],
                                                                 n_sites=config['n_sites'],
                                                                 bin_size=int(
                                                                     config['bin_size_ratio'] * config['n_sites']))
    np.savetxt(args.output, simulated_data, delimiter="\t")
    with open(args.config_file_path, "w+") as yaml_file:
        yaml.dump(config, yaml_file)
