import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scripts.analysis.statistical_analysis import get_p_values


def resampling_based_fdr(data_path: str, n_times: int, intermediate_files_path, output_path,
                         ground_truth_files_path=None):
    data = np.loadtxt(data_path, delimiter="\t")
    n_obs = data.shape[0]
    n_features = data.shape[1]
    group_size = n_obs // 2
    p_values = get_p_values(data=data, group1_indices=list(range(group_size)),
                            group2_indices=list(range(group_size, n_obs)), test_type="t-test")
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    perm_p_values = np.zeros((n_times, n_features))
    for i in range(n_times):
        perm_indices = np.random.permutation(n_obs)
        perm_p_values[i] = get_p_values(data=data[perm_indices], group1_indices=list(range(group_size)),
                                        group2_indices=list(range(group_size, n_obs)), test_type="t-test")
    # avg_fp = np.array([np.mean(np.sum(perm_p_values <= sorted_p_values[i], axis=1)) for i in range(n_features)])
    avg_fp = np.array([np.quantile(np.sum(perm_p_values <= sorted_p_values[i], axis=1), 0.99) for i in range(n_features)])
    fdr_hat_sorted = avg_fp / np.arange(1, n_features + 1)
    fdr_hat_sorted = np.minimum.accumulate(fdr_hat_sorted[::-1])[::-1]
    fdr_hat = np.zeros(n_features)
    fdr_hat[sorted_indices] = fdr_hat_sorted
    df = pd.DataFrame({
        "p_values": p_values,
        "resampling_fdr": fdr_hat
    })
    if ground_truth_files_path is not None:
        ground_truth = pd.read_csv(os.path.join(ground_truth_files_path, os.path.basename(data_path)), sep="\t", header=0)
        df["is_false"] = ground_truth["is_false"]
    df.to_csv(os.path.join(intermediate_files_path, os.path.basename(data_path)), sep="\t", index=False)
    if ground_truth_files_path is not None:
        signif_counts = df.groupby(['is_false']).apply(lambda x: x[x < 0.05].count())
        signif_counts = signif_counts.drop(["is_false"], axis=1)
        n_true_signif = signif_counts.loc[1, 'p_values']
        signif_counts.iloc[0] = signif_counts.iloc[0] / 10000
        signif_counts.iloc[1] = signif_counts.iloc[1] / n_true_signif
        signif_counts.loc[1, 'p_values'] = n_true_signif / 10000
        signif_counts = signif_counts.reset_index()
        signif_counts = signif_counts.melt(id_vars=["is_false"])
        signif_counts['dataset'] = os.path.basename(data_path)
        signif_counts.to_csv(os.path.join(output_path, os.path.basename(data_path)), sep="\t", index=False)
    else:
        signif_counts = df[df["resampling_fdr"] < 0.05].shape[0]
        signif_counts = pd.DataFrame(
            {"num_significant_findings": [signif_counts], "dataset": [os.path.basename(data_path)]})
        signif_counts.to_csv(os.path.join(output_path, os.path.basename(data_path)), sep="\t", index=False)


def compute_resampling_based_fdr(target_datasets_list, rawdata_path, intermediate_files_path, output_path, ground_truth_files_path=None):
    os.makedirs(intermediate_files_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    target_datasets_list = pd.read_csv(target_datasets_list, sep="\t", header=None, index_col=None)[0].to_list()
    target_datasets_files = [os.path.join(rawdata_path, f"id~{dataset}.tsv") for dataset in target_datasets_list]
    pool = Pool(len(target_datasets_files))
    pool.starmap(resampling_based_fdr,
                 [(dataset, 1000, intermediate_files_path, output_path, ground_truth_files_path) for dataset in target_datasets_files])


if __name__ == '__main__':
    rawdata_path = "/path/to/raw/data"
    target_datasets_list = "/path/to/list_of_datasets.tsv"
    intermediate_files_path = "/path/to/intermediate_files"
    output_path = "/path/to/results"
    ground_truth_files_path = "/path/to/ground/truth/false_null_hypotheses"
    compute_resampling_based_fdr(target_datasets_list, rawdata_path, intermediate_files_path, output_path,
                                 ground_truth_files_path)
