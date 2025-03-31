library(DESeq2)

args <- commandArgs(trailingOnly = TRUE)
rawcounts_file <- args[1]
metadata_file <- args[2]
results_path <- args[3]
n_repetitions <- as.numeric(args[4])
seed_offest <- as.numeric(args[5])
output_file <- args[6]


meta_data <- read.table(metadata_file, header=T, sep="\t")
sample_id = meta_data$sample_id

data <- read.table(rawcounts_file, header=T, sep="\t")
rownames(data) <- data[,1]
data <- data[,-1]

common_sample_names <- intersect(sample_id, colnames(data))
data <- data[,common_sample_names]
sample_id <- sample_id[sample_id %in% common_sample_names]

repeat_experiment <- function(n_repetitions, results_path, sample_id, data, seed_offest){
  for (i in 1:n_repetitions) {
    dir.create(paste(results_path, "/", i, "/", sep=""), showWarnings = FALSE, recursive = TRUE)
    seed <- i + seed_offest
    set.seed(seed)
    group <- sample(c(rep("case", 31), rep("control", 31)))
    coldata <- data.frame(sample_id = sample_id, group = group)
    
    dds <- DESeqDataSetFromMatrix(countData = data,
                                  colData = coldata,
                                  design= ~ group)
    dds <- DESeq(dds)
    res <- results(dds, name="group_control_vs_case")
    res_file_path <- paste(results_path, "/", i, "/results.txt", sep="")
    meta_data_file_path <- paste(results_path, "/", i, "/metadata.tsv", sep="")
    write.table(res, file = res_file_path, sep = "\t", row.names = TRUE, quote = FALSE)
    write.table(coldata, file = meta_data_file_path, sep = "\t", row.names = FALSE, quote = FALSE)
  }
}

count_pvalues <- function(results_wd, threshold, output_file) {
  subdirs <- list.dirs(results_wd, full.names = TRUE, recursive = FALSE)
  results <- data.frame(dataset_number = character(), n_significant = integer())
  for (subdir in subdirs) {
    pvals_path <- file.path(subdir, "results.txt")
    pvals <- read.table(pvals_path, header = TRUE, sep = "\t", row.names = 1)
    count_under_threshold <- sum(pvals$padj < threshold, na.rm = TRUE)
    subdir_name <- basename(subdir)
    results <- rbind(results, data.frame(dataset_number = subdir_name, n_significant = count_under_threshold))
  }
  write.table(results, file = output_file, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
}

repeat_experiment(n_repetitions, results_path, sample_id, data, seed_offest)
count_pvalues(results_path, 0.1, output_file)
