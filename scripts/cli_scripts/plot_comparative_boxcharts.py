import argparse
from fdr_hacking.data_generation import *
import pandas as pd
import plotly.express as px
from fdr_hacking.util import parse_yaml_file


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concatenated_results', help='Path to the concatenated results file', required=True)
    parser.add_argument('--reporting_config_file', help='Path to the reporting config file', required=True)
    parser.add_argument('--output_dir', help='Path to the output directory to store the charts', required=True)
    args = parser.parse_args()
    concatenated_results_df = pd.read_csv(args.concatenated_results, sep="\t", header=0, index_col=False)
    reporting_config = parse_yaml_file(args.reporting_config_file)
    reporting_config = reporting_config['reporting']
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for key, val in reporting_config.items():
        filtering_criteria = without(val, 'x_axis')
        filtered_df = concatenated_results_df[concatenated_results_df.apply(lambda row: all(row[column] in condition
                                                                                            for column, condition in
                                                                                            filtering_criteria.items()),
                                                                            axis=1)]
        fig = px.box(filtered_df, x=val['x_axis'], y="num_significant_findings", color="dependencies")
        fig.update_xaxes(type='category')
        fig.write_html(os.path.join(args.output_dir, f"{key}.html"))


def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d
