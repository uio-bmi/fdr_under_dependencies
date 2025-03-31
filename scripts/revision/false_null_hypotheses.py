import os.path
import numpy as np
import pandas as pd
from scripts.analysis.statistical_analysis import get_p_values, adjust_p_values

def reanalyse_dataset_with_false_null_hypotheses(data_path, intermediate_files_path, output_path, modified_data_path):
    data = np.loadtxt(data_path, delimiter="\t")
    n_obs = data.shape[0]
    group_size = n_obs // 2
    data, selected_indices = select_and_modify_features(data, p=200, a=2, b=2.5, z=1)
    np.savetxt(os.path.join(modified_data_path, os.path.basename(data_path)), data, delimiter="\t")
    p_values = get_p_values(data=data, group1_indices=list(range(group_size)),
                            group2_indices=list(range(group_size, n_obs)), test_type="t-test")
    adjustment_methods = ['bonferroni', 'bh', 'by', 'ts_by', "ts_bh", "hs", "h", "s", "sh"]
    is_false = np.zeros(data.shape[1])
    is_false[selected_indices] = 1
    fdr_results = {'p_values': p_values, 'is_false': is_false}
    for method in adjustment_methods:
        fdr_results[f"pdj_{method}"] = adjust_p_values(p_values=p_values, method=method)
    fdr_results = pd.DataFrame(fdr_results)
    fdr_results.to_csv(os.path.join(intermediate_files_path, os.path.basename(data_path)), sep="\t", index=False)
    # group by is_false and method and count number of significant findings
    signif_counts = fdr_results.groupby(['is_false']).apply(lambda x: x[x < 0.05].count())
    signif_counts = signif_counts.drop(["is_false"], axis=1)
    n_true_signif = signif_counts.loc[1, 'p_values']
    signif_counts.iloc[0] = signif_counts.iloc[0] / 10000
    signif_counts.iloc[1] = signif_counts.iloc[1] / n_true_signif
    signif_counts.loc[1, 'p_values'] = n_true_signif / 10000
    signif_counts = signif_counts.reset_index()
    signif_counts = signif_counts.melt(id_vars=["is_false"])
    signif_counts['dataset'] = os.path.basename(data_path)
    signif_counts.to_csv(os.path.join(output_path, os.path.basename(data_path)), sep="\t", index=False)


def select_and_modify_features(data, p, a, b, z):
    """
    Selects 'p' random features from a 2D numpy array whose mean values are
    within a specified range, and adds a constant to the second half of the
    observations for those selected features.

    Args:
        data (numpy.ndarray): A 2D numpy array (n observations x m features).
        p (int): The number of features to select.
        a (float): The lower bound of the mean value range.
        b (float): The upper bound of the mean value range.
        z (float): The constant value to add to the second half of observations.

    Returns:
        numpy.ndarray: The modified numpy array.
    """
    n, m = data.shape
    feature_means = np.mean(data, axis=0)
    eligible_indices = np.where((feature_means >= a) & (feature_means <= b))[0]
    if len(eligible_indices) < p:
        raise ValueError(
            f"Not enough features within the specified mean range. Found: {len(eligible_indices)}, Required: {p}")
    selected_indices = np.random.choice(eligible_indices, size=p, replace=False)
    half_n = n // 2
    data[half_n:, selected_indices] += z
    return data, selected_indices


def reanalyse_datasets(target_datasets_list, rawdata_path, intermediate_files_path, output_path, modified_data_path):
    os.makedirs(intermediate_files_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(modified_data_path, exist_ok=True)
    target_datasets_list = pd.read_csv(target_datasets_list, sep="\t", header=None, index_col=None)[0].to_list()
    target_datasets_files = [os.path.join(rawdata_path, f"id~{dataset}.tsv") for dataset in target_datasets_list]
    for dataset in target_datasets_files:
        print(f"Reanalysing dataset: {dataset}")
        reanalyse_dataset_with_false_null_hypotheses(dataset, intermediate_files_path, output_path, modified_data_path)


if __name__ == '__main__':
    rawdata_path = "/path/to/raw/data"
    target_datasets_list = "/path/to/list_of_datasets.tsv"
    intermediate_files_path = "/path/to/intermediate_files"
    output_path = "/path/to/results"
    modified_data_path = "/path/to/modified_data"
    reanalyse_datasets(target_datasets_list, rawdata_path, intermediate_files_path, output_path, modified_data_path)
