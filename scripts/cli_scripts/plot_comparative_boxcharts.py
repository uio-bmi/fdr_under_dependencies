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
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for key, val in reporting_config.items():
        filtering_criteria = without(val, 'x_axis')
        filtered_df = concatenated_results_df[concatenated_results_df.apply(lambda row: all(row[column] == condition
                                                                                            for column, condition in
                                                                                            filtering_criteria.items()),
                                                                            axis=1)]
        fig = px.box(filtered_df, x=val['x_axis'], y="num_significant_findings", color="dependencies")
        # fig.update_traces(quartilemethod="exclusive")
        fig.update_xaxes(type='category')
        fig.write_html(os.path.join(args.output_dir, f"{key}.html"))


    # config_cols = ["n_observations", "n_sites", "dependencies", "correlation_strength", "bin_size_ratio",
    #                "statistical_test", "multipletest_correction_type", "alpha", "reporting_histogram_bins"]
    # concatenated_results_df['config_id'] = ['__'.join(f'{k}~{v}' for k, v in i.items()) for i in concatenated_results_df[config_cols].to_dict('records')]
    # config_cols.append("config_id")
    # df_grouped = concatenated_results_df.groupby(config_cols)
    # hist_df = pd.DataFrame(columns=config_cols + ["reporting_histogram"])
    # for name, group in df_grouped:
    #     uniq_row = list(name)
    #     hist, bin_edges = np.histogram(a=group['num_significant_findings'],
    #                                    bins=ast.literal_eval(group['reporting_histogram_bins'].iloc[0]))
    #     uniq_row.append(list(hist))
    #     hist_df = pd.concat([hist_df, pd.DataFrame([uniq_row], columns=config_cols + ["reporting_histogram"])],
    #                         ignore_index=True)
    # hist_df.to_csv(args.aggregated_results, sep="\t", index=False)

def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d
