import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scripts.analysis.statistical_analysis import get_p_values, adjust_p_values

raw_data_directory = 'data/raw_without_dependencies'
p_values_directory = 'data/independent_p_values'
adjusted_p_values_directory = 'data/independent_adjusted_p_values'
test_type = "t-test"
method = "bh"


def process_file(file):
    data_path = os.path.join(raw_data_directory, file)
    data = np.loadtxt(data_path, delimiter="\t")
    n_obs = data.shape[0]
    group_size = n_obs // 2
    group1_indices = list(range(group_size))
    group2_indices = list(range(group_size, n_obs))
    p_values = get_p_values(data=data, group1_indices=group1_indices,
                            group2_indices=group2_indices, test_type=test_type)
    np.savetxt(os.path.join(p_values_directory, file), p_values, delimiter="\t")
    adjusted_p_values = adjust_p_values(p_values=p_values, method=method)
    np.savetxt(os.path.join(adjusted_p_values_directory, file), adjusted_p_values, delimiter="\t")

files = os.listdir(raw_data_directory)
with ThreadPoolExecutor(max_workers=12) as executor:
    executor.map(process_file, files)
