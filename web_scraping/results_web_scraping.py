import pandas as pd

output_bmc_genomics = pd.read_csv("results/output_bmc_genomics.csv")
output_bmc_genomics = output_bmc_genomics[output_bmc_genomics["mentions_bh"] + output_bmc_genomics["mentions_bonferroni"] + output_bmc_genomics["mentions_by"] <= 1]
output_clinical_epigenetics = pd.read_csv("results/output_clinical_epigenetics.csv")
output_clinical_epigenetics = output_clinical_epigenetics[output_clinical_epigenetics["mentions_bh"] + output_clinical_epigenetics["mentions_bonferroni"] + output_clinical_epigenetics["mentions_by"] <= 1]
output_genome_biology = pd.read_csv("results/output_genome_biology.csv")
output_genome_biology = output_genome_biology[output_genome_biology["mentions_bh"] + output_genome_biology["mentions_bonferroni"] + output_genome_biology["mentions_by"] <= 1]

print("Ratio of articles mentioning BH in BMC Genomics:", output_bmc_genomics["mentions_bh"].mean())
print("Ratio of articles mentioning BH in Clinical Epigenetics:", output_clinical_epigenetics["mentions_bh"].mean())
print("Ratio of articles mentioning BH in Genome Biology:", output_genome_biology["mentions_bh"].mean())
print("\n")

print("Ratio of articles mentioning Bonferroni in BMC Genomics:", output_bmc_genomics["mentions_bonferroni"].mean())
print("Ratio of articles mentioning Bonferroni in Clinical Epigenetics:", output_clinical_epigenetics["mentions_bonferroni"].mean())
print("Ratio of articles mentioning Bonferroni in Genome Biology:", output_genome_biology["mentions_bonferroni"].mean())
print("\n")

print("Ratio of articles mentioning Benjamini-Yekutieli in BMC Genomics:", output_bmc_genomics["mentions_by"].mean())
print("Ratio of articles mentioning Benjamini-Yekutieli in Clinical Epigenetics:", output_clinical_epigenetics["mentions_by"].mean())
print("Ratio of articles mentioning Benjamini-Yekutieli in Genome Biology:", output_genome_biology["mentions_by"].mean())
print("\n")

results = {
    "BMC Genomics": {
        "BH": output_bmc_genomics["mentions_bh"].mean(),
        "Bonferroni": output_bmc_genomics["mentions_bonferroni"].mean(),
        "BY": output_bmc_genomics["mentions_by"].mean(),
    },
    "Clinical Epigenetics": {
        "BH": output_clinical_epigenetics["mentions_bh"].mean(),
        "Bonferroni": output_clinical_epigenetics["mentions_bonferroni"].mean(),
        "BY": output_clinical_epigenetics["mentions_by"].mean(),
    },
    "Genome Biology": {
        "BH": output_genome_biology["mentions_bh"].mean(),
        "Bonferroni": output_genome_biology["mentions_bonferroni"].mean(),
        "BY": output_genome_biology["mentions_by"].mean(),
    },
}

pd.DataFrame(results).to_csv("results/method_usage_ratios.csv")
