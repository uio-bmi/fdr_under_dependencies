import unittest

import numpy as np

from scripts.analysis.statistical_analysis import perform_t_test


class TestStatisticalAnalysis(unittest.TestCase):

    def setUp(self):
        self.methylation_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        self.group1_indices = [0, 1]
        self.group2_indices = [2, 3]

    def test_perform_t_test(self):
        result = perform_t_test(self.methylation_data, self.group1_indices, self.group2_indices)
        expected_result = np.array([0.105573, 0.105573, 0.105573])
        np.testing.assert_allclose(result, expected_result, rtol=1e-5)

# @pytest.fixture
# def toy_methyl_data():
#     data = np.random.randn(100, 10)  # observations in rows, features in columns
#     group1_indices = list(range(50))
#     group2_indices = list(range(50, 100))
#     return data, group1_indices, group2_indices
#
#
# def test_t_test(toy_methyl_data):
#     p_values = perform_t_test(*toy_methyl_data)
#     assert p_values.shape == (10,)
#     assert (p_values >= 0).all() and (p_values <= 1).all()
#
#
# def test_limma_test(toy_methyl_data):
#     p_values = perform_limma_test(*toy_methyl_data)
#     assert p_values.shape == (10,)
#     assert (p_values >= 0).all() and (p_values <= 1).all()
#
#
# def test_ranksum_test(toy_methyl_data):
#     p_values = perform_t_test(*toy_methyl_data)
#     assert p_values.shape == (10,)
#     assert (p_values >= 0).all() and (p_values <= 1).all()
#
#
# def test_adjust_pvalues():
#     p_values = np.array([0.01, 0.05, 0.1, 0.001, 0.03])
#     adjusted_pvalues = adjust_p_values(p_values, method='bonferroni')
#     expected_adjusted_pvalues = np.array([0.05, 0.25, 0.5, 0.005, 0.15])
#     assert np.allclose(adjusted_pvalues, expected_adjusted_pvalues)
#     p_values = np.array([0.01, 0.05, 0.1, 0.001, 0.03])
#     adjusted_pvalues = adjust_p_values(p_values, method='bh')
#     expected_adjusted_pvalues = np.array([0.025, 0.0625, 0.1, 0.005, 0.05])
#     assert np.allclose(adjusted_pvalues, expected_adjusted_pvalues)
#     with pytest.raises(ValueError):
#         adjust_p_values(p_values, method='invalid_method')
#
#
# def test_count_significant_pvalues():
#     alpha = 0.05
#     adjusted_pvalues = np.array([])
#     num_significant = count_significant_p_values(adjusted_pvalues, alpha)
#     assert num_significant == 0
#     adjusted_pvalues = np.array([0.3, 0.6, 0.8])
#     num_significant = count_significant_p_values(adjusted_pvalues, alpha)
#     assert num_significant == 0
#     adjusted_pvalues = np.array([0.04, 0.003, 0.5, 0.8])
#     num_significant = count_significant_p_values(adjusted_pvalues, alpha)
#     assert num_significant == 2
