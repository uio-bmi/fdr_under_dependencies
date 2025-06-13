import pandas as pd

file_paths = {
    "BMC Genomics": "results/bmc_genomics.csv",
    "Clinical Epigenetics": "results/clinical_epigenetics.csv",
    "Genome Biology": "results/genome_biology.csv",
    "Genome Research": "results/genome_research.csv",
    "NAR": "results/nar.csv",
    "Nature Communications": "results/nature_communications.csv",
    "Nature": "results/nature.csv",
    "Nature Genetics": "results/nature_genetics.csv"
}

results = {}

for journal, path in file_paths.items():
    df = pd.read_csv(path)

    methylation_count = df["mentions_methylation"].sum()
    stat_test_and_correction_count = ((df["mentions_methylation"] == 1) & (df["mentions_stat_test"] == 1) & (df["mentions_multiple_correction"] == 1)).sum()

    df_filtered = df[(df["mentions_methylation"] == 1) & (df["mentions_stat_test"] == 1) & (df["mentions_multiple_correction"] == 1)]
    mentions_count = df_filtered[["mentions_bh", "mentions_bonferroni", "mentions_by"]].sum(axis=1)
    df_single_mention = df_filtered[mentions_count == 1]

    results[journal] = {
        "Benjamini-Hochberg (%)": round((df_single_mention["mentions_bh"].sum() / len(df_filtered)) * 100, 2),
        "Benjamini-Hochberg (Count)": df_single_mention["mentions_bh"].sum(),
        "Bonferroni (%)": round((df_single_mention["mentions_bonferroni"].sum() / len(df_filtered)) * 100, 2),
        "Bonferroni (Count)": df_single_mention["mentions_bonferroni"].sum(),
        "Benjamini-Yekutieli (%)": round((df_single_mention["mentions_by"].sum() / len(df_filtered)) * 100, 2),
        "Benjamini-Yekutieli (Count)": df_single_mention["mentions_by"].sum(),
        "More than one method (%)": round((mentions_count > 1).mean() * 100, 2),
        "More than one method (Count)": (mentions_count > 1).sum(),
        "None (%)": round((mentions_count == 0).mean() * 100, 2),
        "None (Count)": (mentions_count == 0).sum(),
        "Mentions Methylation (Count)": methylation_count,
        "Mentions Stat Test & Correction (Count)": stat_test_and_correction_count
    }

results_df = pd.DataFrame.from_dict(results, orient="index")

pd.DataFrame(results_df).to_csv("results/method_usage_ratios.csv")
