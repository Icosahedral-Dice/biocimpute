# Generate parity fixtures from the reference R package impute.
#
# Usage (from the repo root):
#   tests/r/install_impute.sh
#   Rscript tests/r/generate_fixtures.R
#
# Writes tests/fixtures/manifest.csv (one row per scenario with parameters
# and R's warning / error / console output) plus <name>_input.csv for
# synthetic inputs and <name>_imputed.csv with the imputed values of the cells
# that were missing in the input (1-based row, col; column-major order).
# Observed cells are returned unchanged by impute.knn.
.libPaths(c("tests/r/lib", .libPaths()))
library(impute)

outdir <- "tests/fixtures"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

write_matrix <- function(x, file) {
  s <- matrix(sprintf("%.17g", x), nrow = nrow(x))
  write.table(s, file, sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE)
}

data(khanmiss)
khan <- as.matrix(khanmiss[-1, -(1:2)])
storage.mode(khan) <- "double"
dimnames(khan) <- NULL

write_imputed <- function(x, res, file) {
  idx <- which(is.na(x), arr.ind = TRUE)
  stopifnot(all(res[!is.na(x)] == x[!is.na(x)]))
  out <- data.frame(row = idx[, 1], col = idx[, 2], value = sprintf("%.17g", res[idx]))
  write.csv(out, file, quote = FALSE, row.names = FALSE)
}

sprinkle_na <- function(x, prob) {
  x[matrix(runif(length(x)) < prob, nrow(x))] <- NA
  x
}

set.seed(20261019)
inputs <- list()

inputs$rand_heavy <- sprinkle_na(matrix(rnorm(200 * 20), 200), 0.7)

x <- matrix(rnorm(30 * 10), 30)
x <- sprinkle_na(x, 0.1)
x[7, 1:8] <- NA
inputs$rand_one_row_over_rowmax <- x

x <- matrix(rnorm(40 * 10), 40)
x <- sprinkle_na(x, 0.15)
x[c(5, 12, 13, 30), 2:9] <- NA
inputs$rand_rows_over_rowmax <- x

x <- matrix(sample(0:3, 60 * 6, replace = TRUE), 60)
x <- rbind(x, x[1:20, ])
inputs$dup_ties <- sprinkle_na(x, 0.2)

inputs$rand_split <- sprinkle_na(matrix(rnorm(500 * 8), 500) +
                                   rep(c(-3, 3), each = 250), 0.1)

x <- sprinkle_na(matrix(rnorm(50 * 6), 50), 0.1)
x[, 3] <- NA
inputs$rand_all_na_column <- x

x <- sprinkle_na(matrix(rnorm(20 * 5), 20), 0.1)
x[1:17, 2] <- NA
inputs$rand_colmax_error <- x

scenarios <- list(
  list(name = "khan_default", input = "khan"),
  list(name = "khan_knn_only", input = "khan", maxp = 2308),
  list(name = "khan_k1", input = "khan", k = 1),
  list(name = "khan_k5_seed12345", input = "khan", k = 5, rng.seed = 12345),
  list(name = "khan_negative_seed", input = "khan", maxp = 1000, rng.seed = -5),
  list(name = "khan_maxp200", input = "khan", maxp = 200),
  list(name = "khan_maxp50", input = "khan", maxp = 50),
  list(name = "khan_rowmax", input = "khan", rowmax = 0.02),
  list(name = "khan_colmax_error", input = "khan", colmax = 0.001),
  list(name = "rand_heavy", input = "rand_heavy", k = 3, rowmax = 0.9, colmax = 0.9),
  list(name = "rand_one_row_over_rowmax", input = "rand_one_row_over_rowmax", k = 5),
  list(name = "rand_rows_over_rowmax", input = "rand_rows_over_rowmax", k = 5),
  list(name = "dup_ties", input = "dup_ties", k = 4),
  list(name = "rand_split", input = "rand_split", maxp = 100),
  list(name = "rand_split_small_clusters", input = "rand_split", k = 15, maxp = 30),
  list(name = "rand_all_na_column", input = "rand_all_na_column", k = 5, colmax = 1),
  list(name = "rand_colmax_error", input = "rand_colmax_error")
)

defaults <- list(k = 10, rowmax = 0.5, colmax = 0.8, maxp = 1500, rng.seed = 362436069)

for (nm in names(inputs)) {
  write_matrix(inputs[[nm]], file.path(outdir, paste0(nm, "_input.csv")))
}

rows <- list()
for (sc in scenarios) {
  args <- modifyList(defaults, sc[setdiff(names(sc), c("name", "input"))])
  x <- if (sc$input == "khan") khan else inputs[[sc$input]]
  warn <- NA_character_
  err <- NA_character_
  res <- NULL
  out <- capture.output(
    res <- tryCatch(
      withCallingHandlers(
        do.call(impute.knn, c(list(data = x), args)),
        warning = function(w) {
          warn <<- conditionMessage(w)
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) {
        err <<- conditionMessage(e)
        NULL
      }
    )
  )
  if (!is.null(res)) {
    write_imputed(x, res$data, file.path(outdir, paste0(sc$name, "_imputed.csv")))
  }
  rows[[length(rows) + 1]] <- data.frame(
    name = sc$name, input = sc$input, k = args$k, rowmax = args$rowmax,
    colmax = args$colmax, maxp = args$maxp, rng_seed = args$rng.seed,
    warning = warn, error = err, stdout = paste(out, collapse = "\n"),
    stringsAsFactors = FALSE
  )
}

manifest <- do.call(rbind, rows)
write.csv(manifest, file.path(outdir, "manifest.csv"), row.names = FALSE, na = "")
cat("impute", as.character(packageVersion("impute")), "-",
    nrow(manifest), "scenarios written to", outdir, "\n")
