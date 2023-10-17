import argparse
import json
from fdr_hacking.data_generation import *
import numpy as np
import yaml


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in YAML format', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    parser.add_argument('--config_file_path', help='Path to the simulation config file', required=True)
    args = parser.parse_args()

    config = json.loads(args.config.replace("\'", "\"")) #TODO: fix this
    realworld_data = load_eg_realworld_data()
    if config["dependencies"] == 0:
        simulated_data = simulate_methyl_data(realworld_data, config['n_sites'], config['n_observations'],
                                              config['dependencies'],
                                              None, None)
    else:
        assert isinstance(config['bin_size_ratio'], float), "bin_size_ratio must be a float"
        assert config['correlation_strength'] in ['medium', 'high'], "dependencies must be either medium or high"
        if config['correlation_strength'] == 'medium':
            corr_coef_dist = [(-0.6, -0.4), (-0.1, 0.1), (0.4, 0.6)]
        elif config['correlation_strength'] == 'high':
            corr_coef_dist = [(-0.85, -0.7), (-0.1, 0.1), (0.7, 0.85)]
        simulated_data = simulate_methyl_data(realworld_data, config['n_sites'], config['n_observations'],
                                              config['dependencies'],
                                              int(config['bin_size_ratio'] * config['n_sites']), corr_coef_dist)
    np.savetxt(args.output, simulated_data, delimiter="\t")
    with open(args.config_file_path, "w+") as yaml_file:
        yaml.dump(config, yaml_file)
