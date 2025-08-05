library(ggplot2)
library(data.table)
library(tidyr)
library(ggthemes)
library(dplyr)
library(pheatmap)
library("RColorBrewer")
library(stringr)
library(tibble)
library(readr)
library(scales)
library(grid)
library(viridis)
library(cowplot)
library(egg)
options(scipen=999)

## Raw data was downloaded from "wget https://rest.xialab.ca/api/download/metaboanalyst/human_cachexia.csv" as described in https://www.metaboanalyst.ca/resources/vignettes/Statistical_Analysis_Module.html

data <- read.table("human_cachexia.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
rownames(data) <- data[,1]
group_info <- data[,2]
data <- data[,-c(1,2)]
data <- log2(data+0.0001)
data_scaled <- scale(data)
cor_mat <- cor(data_scaled, method = "spearman")
cor_mat[lower.tri(cor_mat, diag = T)] <- NA
diag(cor_mat) <-NA
quantile((cor_mat), seq(0,1,0.1), na.rm = TRUE)

## perform a standard t-test for each column (feature) against the group_info
t_test_results <- apply(data, 2, function(x) {
  t.test(x[group_info == "cachexic"], x[group_info == "control"])$p.value
})
t_test_results <- p.adjust(t_test_results, method = "fdr")

## permute the group_info and perform the t-test again and store results as "permuted_t_test_results"
set.seed(123)  # for reproducibility
permuted_group_infos_list <- list()
bh_results <- matrix(NA, nrow = ncol(data), ncol = 10000)
by_results <- matrix(NA, nrow = ncol(data), ncol = 10000)
for (i in 1:10000) {
  permuted_group_info <- sample(group_info)
  permuted_group_infos_list[[i]] <- permuted_group_info
  res <- apply(data, 2, function(x) {
    t.test(x[permuted_group_info == "cachexic"], x[permuted_group_info == "control"])$p.value
  })
  bh_results[, i] <- p.adjust(res, method = "BH")
  by_results[, i] <- p.adjust(res, method = "BY")
}

# how many features have a p-value less than 0.05 in each iteration 
n_significant_features_bh <- colSums(bh_results < 0.05)
n_significant_features_by <- colSums(by_results < 0.05)

bin_edges <- c("0%","1-5%", "5-25%", "25-40%", ">40%")
# cut the n_significant_features_bh into bins
n_significant_features_bh_binned <- cut(n_significant_features_bh, 
                                        breaks = c(-1, 0, 5, 25, 40, Inf), 
                                        labels = bin_edges)
n_significant_features_by_binned <- cut(n_significant_features_by, 
                                        breaks = c(-1, 0, 5, 25, 40, Inf), 
                                        labels = bin_edges)
metabolomics_results <- data.frame(n_significant_features_bh, n_significant_features_bh_binned, 
                                   n_significant_features_by, n_significant_features_by_binned)
