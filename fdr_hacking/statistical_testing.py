import numpy as np
import rpy2.robjects as robjects
from scipy.stats import ttest_ind, ranksums
from statsmodels.stats.multitest import multipletests


def t_test(methyl_datamat: np.ndarray, group1_indices: list, group2_indices: list) -> np.array:
    """
    :param methyl_datamat: A ndimensional numpy array of methylation M values with observations in rows,
    features in columns
    :param group1_indices: A list of indices of first group for the purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :return: A numpy array of p-values with size the same as the number of features in input dataset
    """
    group1_data = methyl_datamat[group1_indices]
    group2_data = methyl_datamat[group2_indices]
    p_values = np.zeros(methyl_datamat.shape[1])
    for col in range(methyl_datamat.shape[1]):
        p_values[col] = ttest_ind(group1_data[:, col], group2_data[:, col])[1]
    return p_values


def limma_test(methyl_datamat: np.ndarray, group1_indices: list, group2_indices: list) -> np.array:
    """

    :param methyl_datamat: A ndimensional numpy array of methylation M values with observations in rows,
    features in columns
    :param group1_indices: A list of indices of first group for the purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :return: A numpy array of p-values with size the same as the number of features in input dataset
    """
    methyl_datamat = methyl_datamat.T
    r_data = robjects.r['matrix'](robjects.FloatVector(methyl_datamat.flatten()),
                                  nrow=methyl_datamat.shape[0], byrow=True)
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
    pvals = np.array(robjects.r('pvals'))
    return pvals


def ranksum_test(methyl_datamat: np.ndarray, group1_indices: list, group2_indices: list) -> np.array:
    """

    :param methyl_datamat: A ndimensional numpy array of methylation M values with observations in rows,
    features in columns
    :param group1_indices: A list of indices of first group for the purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :return: A numpy array of p-values with size the same as the number of features in input dataset
    """
    group1_data = methyl_datamat[group1_indices]
    group2_data = methyl_datamat[group2_indices]
    p_values = np.zeros(methyl_datamat.shape[1])
    for col in range(methyl_datamat.shape[1]):
        p_values[col] = ranksums(group1_data[:, col], group2_data[:, col])[1]
    return p_values


def get_p_values_per_feature(methyl_datamat: np.ndarray, group1_indices: list, group2_indices: list,
                             test_type: str) -> np.array:
    """

    :param methyl_datamat: A ndimensional numpy array of methylation M values with observations in rows,
    features in columns
    :param group1_indices: A list of indices of first group for the purpose of two-sample statistical testing
    :param group2_indices: A list of indices of second group for the purpose of two-sample statistical testing
    :param test_type: A string indicating the statistical test to use. Legal values are 't-test', 'limma', 'rank-sum'
    :return: A numpy array of p-values with size the same as the number of features in input dataset
    """
    if test_type == 't-test':
        p_values = t_test(methyl_datamat, group1_indices, group2_indices)
    elif test_type == 'limma':
        p_values = limma_test(methyl_datamat, group1_indices, group2_indices)
    elif test_type == 'rank-sum':
        p_values = ranksum_test(methyl_datamat, group1_indices, group2_indices)
    else:
        raise ValueError("Invalid test type")
    return p_values


def adjust_pvalues(p_values: np.array, method: str) -> np.array:
    """

    :param p_values: A numpy array of p-values
    :param method: A string indicating the method to use for adjusting p-values for multiple testing. Legal values are
    "bonferroni" or "bh" (Benjamini-Hochberg)
    :return: A numpy array of p-values corrected for multiple testing using any of the selected methods
    """
    if method == "bonferroni":
        reject, adjusted_pvalues, _, _ = multipletests(p_values, method='bonferroni')
    elif method == "bh":
        reject, adjusted_pvalues, _, _ = multipletests(p_values, method='fdr_bh')
    else:
        raise ValueError("Invalid method specified. Method must be either 'bonferroni' or 'bh'.")
    return adjusted_pvalues


def count_significant_pvalues(adjusted_pvalues: np.array, alpha: float) -> int:
    """

    :param adjusted_pvalues: A numpy array of p-values adjusted for multiple testing
    :param alpha: A float of pre-specified significance level
    :return: An integer indicating the number of significant findings at the pre-specified significance level
    """
    num_significant = np.sum(adjusted_pvalues < alpha)
    return num_significant


def quantify_fdr(methyl_datamat: np.ndarray, group1_indices: list, group2_indices: list, test_type: str, method: str,
                 alpha: float):
    pvals = get_p_values_per_feature(methyl_datamat=methyl_datamat, group1_indices=group1_indices,
                                     group2_indices=group2_indices, test_type=test_type)
    adjusted_pvals = adjust_pvalues(p_values=pvals, method=method)
    num_significant = count_significant_pvalues(adjusted_pvalues=adjusted_pvals, alpha=alpha)
    return num_significant
