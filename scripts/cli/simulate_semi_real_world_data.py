import argparse
import json

import numpy as np
import yaml

from scripts.analysis.data_generation import load_realworld_data, sample_realworld_methyl_val, beta_to_m


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in YAML format', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    parser.add_argument('--config_file_path', help='Path to the simulation config file', required=True)
    parser.add_argument('--realworld_data_path', help='Path to the real-world dataset in H5 file format',
                        required=True)
    args = parser.parse_args()
    config = json.loads(args.config.replace("\'", "\""))  # TODO: fix this
    real_world_data = load_realworld_data(file_path=args.realworld_data_path)
    semi_real_world_data = sample_realworld_methyl_val(n_sites=config['n_sites'], realworld_data=real_world_data)
    semi_real_world_data = beta_to_m(methyl_beta_values=semi_real_world_data)
    np.random.shuffle(semi_real_world_data)
    np.savetxt(args.output, semi_real_world_data, delimiter="\t")
    with open(args.config_file_path, "w+") as yaml_file:
        yaml.dump(config, yaml_file)
