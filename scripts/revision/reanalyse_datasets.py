import os.path
import numpy as np
import pandas as pd
from scripts.analysis.statistical_analysis import get_p_values, adjust_p_values


def reanalyse_dataset(data_path, intermediate_files_path, output_path):
    data = np.loadtxt(data_path, delimiter="\t")
    n_obs = data.shape[0]
    group_size = n_obs // 2
    p_values = get_p_values(data=data, group1_indices=list(range(group_size)),
                            group2_indices=list(range(group_size, n_obs)), test_type="t-test")
    adjustment_methods = ['bonferroni', 'bh', 'by', 'ts_by']
    fdr_results = {'p_values': p_values}
    for method in adjustment_methods:
        fdr_results[f"pdj_{method}"] = adjust_p_values(p_values=p_values, method=method)
    fdr_results = pd.DataFrame(fdr_results)
    fdr_results.to_csv(os.path.join(intermediate_files_path, os.path.basename(data_path)), sep="\t", index=False)

    signif_counts = fdr_results[fdr_results < 0.05].count()
    signif_counts = pd.DataFrame(signif_counts).T
    signif_counts['dataset'] = os.path.basename(data_path)
    signif_counts.to_csv(os.path.join(output_path, os.path.basename(data_path)), sep="\t", index=False)

def reanalyse_datasets(target_datasets_list, rawdata_path, intermediate_files_path, output_path):
    os.makedirs(intermediate_files_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    target_datasets_list = pd.read_csv(target_datasets_list, sep="\t", header=None, index_col=None)[0].to_list()
    target_datasets_files = [os.path.join(rawdata_path, f"id~{dataset}.tsv") for dataset in target_datasets_list]
    for dataset in target_datasets_files:
        print(f"Reanalysing dataset: {dataset}")
        reanalyse_dataset(dataset, intermediate_files_path, output_path)


if __name__ == '__main__':
    rawdata_path = "/path/to/raw/data"
    target_datasets_list = "/path/to/list_of_datasets.tsv"
    intermediate_files_path = "/path/to/intermediate_files"
    output_path = "/path/to/results"
    reanalyse_datasets(target_datasets_list, rawdata_path, intermediate_files_path, output_path)
