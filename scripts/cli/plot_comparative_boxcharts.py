import argparse
import os

import pandas as pd
import plotly.express as px

from scripts.analysis.utils import parse_yaml_file


def parse_args():
    """
    This function parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--concatenated_results', help='Path to the concatenated results file', required=True)
    parser.add_argument('--reporting_config_file', help='Path to the reporting config file', required=True)
    parser.add_argument('--output_dir', help='Path to the output directory to store the charts', required=True)
    args = parser.parse_args()

    return args


def main(concatenated_results, reporting_config_file, output_dir):
    """
    This function plots the boxcharts for the concatenated results.

    :param concatenated_results: Path to the concatenated results file in tsv format
    :param reporting_config_file: Path to the reporting config file in yaml format
    :param output_dir: Path to the output directory to store the charts
    :return: None
    """
    concatenated_results_df = pd.read_csv(concatenated_results, sep="\t", header=0, index_col=False)
    reporting_config = parse_yaml_file(reporting_config_file)['reporting']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for key, val in reporting_config.items():
        filtering_criteria = without(val, 'x_axis')
        filtered_df = concatenated_results_df[concatenated_results_df.apply(lambda row:
                                                                            all(row[column] in condition
                                                                                for column, condition in
                                                                                filtering_criteria.items()), axis=1)]

        sort_ascending = val['x_axis'] != "correlation_strength"
        filtered_df = filtered_df.sort_values(by=val['x_axis'], ascending=sort_ascending)

        color_discrete_map = {0: '#4682B4', 1: '#E97451'}
        if "dependencies" in filtered_df.columns:
            filtered_df.loc[:, 'dependencies'] = filtered_df['dependencies'].astype(bool)
            fig = px.box(filtered_df, x=val['x_axis'], y="num_significant_findings", color="dependencies",
                         color_discrete_map=color_discrete_map)
        else:
            fig = px.box(filtered_df, x=val['x_axis'], y="num_significant_findings")

        y_title = "Number of significant findings"
        x_title_map = {"alpha": "Alpha", "multipletest_correction_type": "Multiple testing correction",
                       "statistical_test": "Statistical test", "bin_size_ratio": "Bin size ratio",
                       "correlation_strength": "Correlation strength", "n_sites": "Number of sites",
                       "n_observations": "Number of observations"}
        fig.update_xaxes(type='category')
        fig.update_layout(
            legend=dict(font=dict(size=18)),
            xaxis=dict(title=dict(text=x_title_map[val['x_axis']], font=dict(size=22)), tickfont=dict(size=18)),
            yaxis=dict(title=dict(text=y_title, font=dict(size=22)), tickfont=dict(size=18), range=[0, 2500])
        )

        fig.write_html(os.path.join(output_dir, f"{key}.html"))

        png_file_path = os.path.join(output_dir, f"{key}.png")
        fig.write_image(png_file_path, height=900, width=900)


def without(dictionary, key):
    """
    This function returns a copy of the dictionary without the specified key.

    :param dictionary: Dictionary
    :param key: Key to remove
    """
    new_dictionary = dictionary.copy()
    new_dictionary.pop(key)
    return new_dictionary


def execute():
    """
    This function is executed when this file is run as a script.
    """
    args = parse_args()
    main(args.concatenated_results, args.reporting_config_file, args.output_dir)
