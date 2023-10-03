import argparse
import json
from fdr_hacking.data_generation import *
import numpy as np


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in YAML format', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    args = parser.parse_args()

    config = json.loads(args.config.replace("\'", "\"")) #TODO: fix this
    realworld_data = load_eg_realworld_data()
    simulated_data = simulate_methyl_data(realworld_data, config['n_sites'], config['n_observations'],
                                          config['dependencies'],
                                          int(config['bin_size_ratio']*config['n_sites']), [(-0.85, -0.50), (-0.1, 0.1), (0.70, 0.85)])
    np.savetxt(args.output, simulated_data, delimiter="\t")
