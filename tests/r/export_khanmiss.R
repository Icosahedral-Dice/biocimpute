# Export impute's khanmiss data.frame to src/biocimpute/datasets/khanmiss.csv.gz.
# Usage (from the repo root): Rscript tests/r/export_khanmiss.R
.libPaths(c("tests/r/lib", .libPaths()))
data(khanmiss, package = "impute")
out <- gzfile("src/biocimpute/datasets/khanmiss.csv.gz", "w")
# All columns are factors; no entry is the literal string "NA".
write.csv(khanmiss, out, row.names = FALSE, na = "NA")
close(out)
