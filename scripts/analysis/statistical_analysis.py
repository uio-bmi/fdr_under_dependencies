import numpy as np
import rpy2.robjects as robjects
from scipy.stats import ttest_ind, ranksums, kstest, norm
from statsmodels.stats.multitest import multipletests


def perform_t_test(methylation_data: np.ndarray, group1_indices: list, group2_indices: list) -> np.array:
    """
    Perform two-sample t-test on each feature in the input dataset
    :param methylation_data: A 2-dimensional numpy array of methylation M values with observations in rows and
    features in columns
    :param group1_indices: A list of indices of first group for the purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :return: A numpy array of p-values
    """
    group1_data = methylation_data[group1_indices]
    group2_data = methylation_data[group2_indices]
    p_values = np.zeros(methylation_data.shape[1])
    for col in range(methylation_data.shape[1]):
        p_values[col] = ttest_ind(group1_data[:, col], group2_data[:, col])[1]
    return p_values


def perform_limma_test(methylation_data: np.ndarray, group1_indices: list, group2_indices: list) -> np.array:
    """
    Perform two-sample limma test on each feature in the input dataset. Since limma is not available in SciPy,
    we use R implementation with Rpy2.
    :param methylation_data: A 2-dimensional numpy array of methylation M values with observations in rows and features
    in columns
    :param group1_indices: A list of indices of first group for the
    purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :return: A numpy array of p-values
    """
    methylation_data_transposed = methylation_data.T
    r_data = robjects.r['matrix'](robjects.FloatVector(methylation_data_transposed.flatten()),
                                  nrow=methylation_data_transposed.shape[0], byrow=True)
    r_group1 = robjects.IntVector(group1_indices)
    r_group2 = robjects.IntVector(group2_indices)
    robjects.r('library(limma)')
    robjects.globalenv['data'] = r_data
    robjects.globalenv['group1'] = r_group1
    robjects.globalenv['group2'] = r_group2
    robjects.r('group <- factor(c(rep("group1", length(group1)), rep("group2", length(group2))))')
    robjects.r('design <- model.matrix(~0+group)')
    robjects.r('colnames(design) <- c("group1", "group2")')
    robjects.r('fit <- lmFit(data, design)')
    robjects.r('contrast.matrix <- makeContrasts(group2-group1, levels=design)')
    robjects.r('fit2 <- contrasts.fit(fit, contrast.matrix)')
    robjects.r('fit2 <- eBayes(fit2)')
    robjects.r('pvals <- topTable(fit2, coef=1, n=Inf)$P.Value')
    p_values = np.array(robjects.r('pvals'))
    return p_values


def perform_rank_sum_test(methylation_data: np.ndarray, group1_indices: list, group2_indices: list) -> np.array:
    """
    Perform Wilcoxon rank-sum test on each feature in the input dataset
    rm two-sample rank-sum test on each feature in the input dataset
    :param methylation_data: A 2-dimensional numpy array of methylation M values with observations in rows and
    features in columns
    :param group1_indices: A list of indices of first group for the purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :return: A numpy array of p-values
    """
    group1_data = methylation_data[group1_indices]
    group2_data = methylation_data[group2_indices]
    p_values = np.zeros(methylation_data.shape[1])
    for col in range(methylation_data.shape[1]):
        p_values[col] = ranksums(group1_data[:, col], group2_data[:, col])[1]
    return p_values


def perform_ks_test(data: np.ndarray) -> np.array:
    """
    Perform Kolmogorov-Smirnov test on each feature in the input dataset
    :param data: A 2-dimensional numpy array of methylation M values with observations in rows and
    features in columns
    :return: A numpy array of p-values
    """
    p_values = np.zeros(data.shape[1])
    for col in range(data.shape[1]):
        p_values[col] = kstest(data[:, col], norm.cdf)[1]
    return p_values


def get_p_values(data: np.ndarray, group1_indices: list, group2_indices: list,
                 test_type: str) -> np.array:
    """
    Perform a two-sample statistical test on each feature in the input dataset
    :param data: A 2-dimensional numpy array with observations in rows and features in columns
    :param group1_indices: A list of indices of first group for the purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :param test_type: A string indicating the statistical test to use. Legal values are 't-test', 'limma', 'rank-sum', 'ks-test'
    :return: A numpy array of p-values
    """
    if test_type == 't-test':
        p_values = perform_t_test(data, group1_indices, group2_indices)
    elif test_type == 'limma':
        p_values = perform_limma_test(data, group1_indices, group2_indices)
    elif test_type == 'rank-sum':
        p_values = perform_rank_sum_test(data, group1_indices, group2_indices)
    elif test_type == 'ks-test':
        p_values = perform_ks_test(data)
    else:
        raise ValueError(f"Invalid test type: {test_type}")
    return p_values


def adjust_p_values(p_values: np.array, method: str) -> np.array:
    """
    Adjust p-values for multiple testing
    :param p_values: A numpy array of p-values
    :param method: A string indicating the method to use for adjusting p-values for multiple testing. Legal values are
    'bonferroni' or 'bh' (Benjamini-Hochberg)
    :return: A numpy array of p-values corrected for multiple testing using any of the selected methods
    """
    correction_methods_map = {
        'bonferroni': 'bonferroni',
        'bh': 'fdr_bh',
    }
    if method in correction_methods_map:
        reject, adjusted_p_values, _, _ = multipletests(p_values, method=correction_methods_map[method])
    else:
        raise ValueError(f"Invalid adjustment method: {method}")
    return adjusted_p_values


def count_significant_p_values(adjusted_p_values: np.array, alpha: float) -> int:
    """
    Count the number of significant findings at the pre-specified significance level
    :param adjusted_p_values: A numpy array of p-values adjusted for multiple testing
    :param alpha: A float of pre-specified significance level
    :return: An integer indicating the number of significant findings at the pre-specified significance level
    """
    significant_findings_number = np.sum(adjusted_p_values < alpha)
    return significant_findings_number


def quantify_significance(data: np.ndarray, group1_indices: list, group2_indices: list, test_type: str, method: str,
                          alpha: float):
    """
    Quantify the number of significant findings at the pre-specified significance level
    :param data: A 2-dimensional numpy array with observations in rows and features in columns
    :param group1_indices: A list of indices of first group for the purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :param test_type: A string indicating the statistical test to use. Legal values are 't-test', 'limma', 'rank-sum'
    :param method: A string indicating the method to use for adjusting p-values for multiple testing. Legal values are 'bonferroni' or 'bh' (Benjamini-Hochberg)
    :param alpha: A float of pre-specified significance level
    :return: An integer indicating the number of significant findings at the pre-specified significance level
    """
    p_values = get_p_values(data=data, group1_indices=group1_indices,
                            group2_indices=group2_indices, test_type=test_type)
    adjusted_p_values = adjust_p_values(p_values=p_values, method=method)
    significant_findings_number = count_significant_p_values(adjusted_p_values=adjusted_p_values, alpha=alpha)
    return significant_findings_number
