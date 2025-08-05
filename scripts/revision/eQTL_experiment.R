## The data and analysis procedures used in this experiment are obtained from the public GitHub repository: https://github.com/perishky/matrixeqtl-tutorial
## The aforementioned tutorial uses publicly available data from this article: http://www.ncbi.nlm.nih.gov/pubmed/24555846

## Covariate information on population stratification has been generated as described in the aforementioned tutorial using PLINK.

library(MatrixEQTL)
library(dplyr)

snp.data <- SlicedData$new()
snp.data$fileDelimiter <- ","
snp.data$fileOmitCharacters <- "NA"
snp.data$fileSkipRows <- 1
snp.data$fileSkipColumns <- 1
snp.data$fileSliceSize <- 2000
snp.data$LoadFile("snp-data.csv")
out.dir <- "output"
snp.mds <- read.table(file.path(out.dir, "plink.mds"), header=T, sep="", stringsAsFactors=F)
covariates.data <- SlicedData$new()
covariates.data$initialize(t(snp.mds[,c("C1","C2")]))
snp.loc <- read.csv("snp-features.csv", row.names=1)
rna.loc <- read.csv("rna-features.csv", row.names=1)

make_rna_object <- function(i) {
  rna.data <- SlicedData$new()
  rna.data$fileDelimiter <- ","
  rna.data$fileOmitCharacters <- "NA"
  rna.data$fileSkipRows <- 1
  rna.data$fileSkipColumns <- 1
  rna.data$fileSliceSize <- 2000
  rna.data$LoadFile(paste0("shuffled_rna/rna_data_shuffled_", i, ".csv"))
}

rna <- read.csv("rna-data.csv", row.names=1)
for(i in 1:10000) {
  rna_data_shuffled <- apply(rna, 1, function(x) sample(x)) %>% t()
  colnames(rna_data_shuffled) <- colnames(rna)
  filename <- file.path("shuffled_rna/", paste0("rna_data_shuffled_", i, ".csv"))
  write.table(rna_data_shuffled,
              file = filename,
              sep = ",", row.names = TRUE, col.names = TRUE, quote = FALSE)
  rna.data <- make_rna_object(i)
  stopifnot(all(colnames(snp.data) == colnames(rna.data)))
  eqtl <- Matrix_eQTL_main(
    snps = snp.data,
    gene = rna.data,
    cvrt = covariates.data, ## population stratification
    pvOutputThreshold = 0, ## consider only cis pairs
    pvOutputThreshold.cis = 0.05,
    output_file_name.cis = file.path(out.dir, paste0("repetitions/eqtl-cis_results_", i, ".txt")),
    snpspos = snp.loc,
    genepos = rna.loc[,c("geneid","chr","left","right")],
    cisDist = 1e6, ## define cis as < 1Mb
    useModel = modelLINEAR, ## test using linear models
    verbose = TRUE,
    pvalue.hist = TRUE,
    min.pv.by.genesnp = FALSE,
    noFDRsaveMemory = FALSE)
  eqtls <- eqtl$cis$eqtls
  significant_eqtls <- eqtls[eqtls$FDR <= 0.05, ]
  total_tests <- eqtl$cis$ntests
  significant_tests <- length(which(eqtls$FDR<=0.05))
  write.table(data.frame(total_tests = total_tests, significant_tests = significant_tests),
              file = file.path(out.dir, paste0("repetition_stats/eqtl-cis_stats_", i, ".txt")),
              sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

}

