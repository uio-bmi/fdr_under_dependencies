import argparse
from fdr_hacking.data_generation import *
from fdr_hacking.util import parse_yaml_file
import numpy as np


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration in YAML format', required=True)
    parser.add_argument('--output', help='Path to the output directory, where all the results should be stored', required=True)
    args = parser.parse_args()
    config = parse_yaml_file(args.config)

