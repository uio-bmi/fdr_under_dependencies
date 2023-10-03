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
    config = json.loads(args.config.replace("\'", "\"")) #TODO: fix this
    data = np.loadtxt(args.dataset, delimiter="\t")
    n_obs = data.shape[0]
    group_size = n_obs // 2
    num_significant_findings = quantify_fdr(methyl_datamat=data,
                                            group1_indices = list(range(group_size)),
                                            group2_indices = list(range(group_size, n_obs)),
                                            test_type=config['statistical_test'],
                                            method=config['multipletest_correction_type'],
                                            alpha=config['alpha'])
    config['num_significant_findings'] = num_significant_findings
    config_df = pd.DataFrame(config, index=[0])
    config_df.to_csv(args.output, sep="\t", index=False)
