import argparse
from fdr_hacking.statistical_testing import *
import numpy as np
import json
import pandas as pd


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration in YAML format', required=True)
    parser.add_argument('--dataset', help='Path to a tab-delimited dataset in the form of a design matrix', required=True)
    parser.add_argument('--output', help='Path to the output file', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.replace("\'", "\""))
    data = np.loadtxt(args.dataset, delimiter="\t")
    n_obs = data.shape[0]
    group_size = n_obs // 2
    num_significant_findings = quantify_fdr(methyl_datamat=data,
                                            group1_indices = list(range(group_size)),
                                            group2_indices = list(range(group_size, n_obs)),
                                            test_type=config['statistical_test'],
                                            method=config['multiple_test_correction'],
                                            alpha=config['alpha'])
    config['num_significant_findings'] = num_significant_findings
    pd.DataFrame(config).to_csv(args.output, sep="\t", index=True, header=True)
    # np.savetxt(args.output, np.array([num_significant_findings], dtype=np.int64), fmt="%d", delimiter="\t")
