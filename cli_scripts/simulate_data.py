import argparse
from fdr_hacking.data_generation import *
from fdr_hacking.util import parse_yaml_file
import numpy as np


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration in YAML format', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    args = parser.parse_args()
    config = parse_yaml_file(args.config)
    realworld_data = load_eg_realworld_data()
    simulated_data = simulate_methyl_data(realworld_data, config['n_sites'], config['n_observations'],
                                          config['dependencies'],
                                          config['bin_size'], config['corr_coef_distribution'])
    np.savetxt(args.output, simulated_data, delimiter="\t")
