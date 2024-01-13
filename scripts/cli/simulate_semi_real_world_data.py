import argparse
import ast

import numpy as np
import yaml

from scripts.analysis.data_generation import load_realworld_data, sample_realworld_methylation_values, beta_to_m


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
    This function generates semi-real-world data and saves it to a file.

    :param config: Path to the configuration file in yaml format
    :param output: Path to the output file
    :param config_file_path: Path to the yaml file where specific parameters for generating data should be saved
    :param realworld_data_path: Path to the real-world dataset in H5 file format
    :return: None
    """
    config = ast.literal_eval(config)
    real_world_data = load_realworld_data(file_path=realworld_data_path)
    semi_real_world_data = sample_realworld_methylation_values(n_sites=config['n_sites'], realworld_data=real_world_data)
    semi_real_world_data = beta_to_m(methyl_beta_values=semi_real_world_data)
    np.random.shuffle(semi_real_world_data)
    np.savetxt(output, semi_real_world_data, delimiter="\t")
    with open(config_file_path, "w+") as yaml_file:
        yaml.dump(config, yaml_file)


def execute():
    """
    This function is executed when this file is run as a script.
    """
    args = parse_args()
    main(args.config, args.output, args.config_file_path, args.realworld_data_path)
