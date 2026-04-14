#!/usr/bin/env Rscript
# Construct real temporal version pairs from Pairwise70 .rda files.
#
# Rationale: even when a review has only one published pub, its study-level
# data carries Study.year. A v1 "snapshot" is the pool of studies with
# Study.year <= cutoff_v1; v2 is the pool with Study.year <= cutoff_v2. Flip
# label comes from whether the CI-crosses-null status changed. This is a
# real retrospective flip signal grounded in actual Cochrane study dates.
#
# For each analysis:
#   - require >=4 studies with non-missing Study.year
#   - place two cutoffs at the 55th and 80th percentiles of study years
#   - require >=2 studies before each cutoff and >=1 more study between
#   - pool each subset via metafor DL random-effects
#
# Output schema matches evidence_forecast.calibration.label_flips.
suppressPackageStartupMessages({
  library(metafor)
})

DATA_DIR <- "C:/Projects/Pairwise70/data"
OUT <- "C:/Models/EvidenceForecast/tests/fixtures/temporal_cochrane_pairs_v0.csv"

pool_studies <- function(df) {
  ok_binary <- nrow(df) >= 2 &&
    all(!is.na(df$Experimental.cases)) && all(!is.na(df$Experimental.N)) &&
    all(!is.na(df$Control.cases))      && all(!is.na(df$Control.N))
  ok_giv    <- nrow(df) >= 2 &&
    all(!is.na(df$GIV.Mean)) && all(!is.na(df$GIV.SE)) &&
    all(df$GIV.SE > 0)
  if (ok_binary) {
    tryCatch({
      es <- escalc(measure = "RR",
                   ai = df$Experimental.cases, n1i = df$Experimental.N,
                   ci = df$Control.cases,      n2i = df$Control.N)
      fit <- rma(yi = es$yi, vi = es$vi, method = "DL")
      list(scale = "RR",
           point = exp(fit$b[1, 1]),
           ci_low = exp(fit$ci.lb),
           ci_high = exp(fit$ci.ub),
           k = fit$k, tau2 = fit$tau2, i2 = fit$I2 / 100)
    }, error = function(e) NULL)
  } else if (ok_giv) {
    tryCatch({
      fit <- rma(yi = df$GIV.Mean, sei = df$GIV.SE, method = "DL")
      list(scale = "MD",
           point = fit$b[1, 1],
           ci_low = fit$ci.lb,
           ci_high = fit$ci.ub,
           k = fit$k, tau2 = fit$tau2, i2 = fit$I2 / 100)
    }, error = function(e) NULL)
  } else NULL
}

files <- list.files(DATA_DIR, pattern = "^CD[0-9]+_pub[0-9]+_data\\.rda$", full.names = TRUE)
cat("RDA files:", length(files), "\n")

rows <- list()
row_idx <- 0

for (f in files) {
  env <- new.env()
  load(f, envir = env)
  obj_name <- ls(env)[1]
  x <- get(obj_name, envir = env)
  review_pub <- sub("_data$", "", obj_name)
  review <- sub("_pub.*$", "", review_pub)
  if (!all(c("Analysis.number", "Analysis.name", "Study.year") %in% colnames(x))) next
  for (anum in unique(x$Analysis.number)) {
    sub <- x[x$Analysis.number == anum & !is.na(x$Study.year), ]
    if (nrow(sub) < 4) next
    outcome_name <- sub$Analysis.name[1]
    years <- sort(unique(sub$Study.year))
    if (length(years) < 3) next
    # Candidate cutoff pairs at 55th and 80th percentiles; clamp to inner years.
    q <- quantile(sub$Study.year, probs = c(0.55, 0.80), type = 1)
    cut1 <- q[1]; cut2 <- q[2]
    if (cut2 <= cut1) next
    v1 <- sub[sub$Study.year <= cut1, ]
    v2 <- sub[sub$Study.year <= cut2, ]
    if (nrow(v1) < 2 || nrow(v2) < nrow(v1) + 1) next
    p1 <- pool_studies(v1); p2 <- pool_studies(v2)
    if (is.null(p1) || is.null(p2)) next
    if (p1$scale != p2$scale) next
    if (is.na(p1$point) || is.na(p2$point)) next
    # Filter any pair where v1 or v2 degenerates (zero-width CI etc.)
    if (!is.finite(p1$ci_low) || !is.finite(p1$ci_high)) next
    if (!is.finite(p2$ci_low) || !is.finite(p2$ci_high)) next
    row_idx <- row_idx + 1
    rows[[row_idx]] <- data.frame(
      ma_id     = sprintf("%s_A%s", review, anum),
      v1_date   = sprintf("%d-06-01", as.integer(cut1)),
      v2_date   = sprintf("%d-06-01", as.integer(cut2)),
      outcome   = outcome_name,
      v1_point  = p1$point,
      v1_ci_low = p1$ci_low,
      v1_ci_high = p1$ci_high,
      v2_point  = p2$point,
      v2_ci_low = p2$ci_low,
      v2_ci_high = p2$ci_high,
      topic_area = "cochrane",
      scale     = p1$scale,
      v1_k      = p1$k,
      v1_tau2   = p1$tau2,
      v1_i2     = p1$i2,
      stringsAsFactors = FALSE
    )
  }
}

cat("generated temporal pairs:", length(rows), "\n")
if (length(rows) == 0) stop("no pairs generated")
out <- do.call(rbind, rows)
write.csv(out, OUT, row.names = FALSE)
cat("wrote:", OUT, "\n")
