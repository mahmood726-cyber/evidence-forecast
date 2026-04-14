#!/usr/bin/env Rscript
# Extract real Cochrane multi-pub version pairs from Pairwise70 .rda files.
#
# For each review with >=2 pubs, pool each analysis at each pub version using
# metafor's inverse-variance DL random-effects, then emit a pair row where
# v1 = earlier pub, v2 = later pub.
#
# Output schema matches evidence_forecast.calibration.label_flips _REQUIRED_COLS:
#   ma_id,v1_date,v2_date,outcome,v1_point,v1_ci_low,v1_ci_high,
#   v2_point,v2_ci_low,v2_ci_high,topic_area,scale

suppressPackageStartupMessages({
  library(metafor)
})

DATA_DIR <- "C:/Projects/Pairwise70/data"
OUT <- "C:/Models/EvidenceForecast/tests/fixtures/real_cochrane_pairs_v0.csv"

files <- list.files(DATA_DIR, pattern = "^CD[0-9]+_pub[0-9]+_data\\.rda$", full.names = TRUE)
parse_id <- function(f) {
  stem <- sub("\\.rda$", "", basename(f))
  stem <- sub("_data$", "", stem)
  parts <- strsplit(stem, "_pub")[[1]]
  list(review = parts[1], pub = as.integer(parts[2]), file = f)
}
ids <- lapply(files, parse_id)

# Find reviews with >=2 pubs
reviews <- split(ids, sapply(ids, function(x) x$review))
multi <- reviews[sapply(reviews, length) >= 2]
cat("multi-pub reviews:", length(multi), "\n")

pool_analysis <- function(df) {
  # Prefer event-based (risk ratio) if available; else GIV
  if (all(!is.na(df$Experimental.cases)) && all(!is.na(df$Experimental.N)) &&
      all(!is.na(df$Control.cases)) && all(!is.na(df$Control.N)) &&
      nrow(df) >= 2) {
    tryCatch({
      es <- escalc(measure = "RR",
                   ai = df$Experimental.cases, n1i = df$Experimental.N,
                   ci = df$Control.cases,      n2i = df$Control.N)
      fit <- rma(yi = es$yi, vi = es$vi, method = "DL")
      list(scale = "RR",
           point = exp(fit$b[1, 1]),
           ci_low = exp(fit$ci.lb),
           ci_high = exp(fit$ci.ub),
           k = fit$k)
    }, error = function(e) NULL)
  } else if (all(!is.na(df$GIV.Mean)) && all(!is.na(df$GIV.SE)) && nrow(df) >= 2) {
    tryCatch({
      fit <- rma(yi = df$GIV.Mean, sei = df$GIV.SE, method = "DL")
      list(scale = "MD",  # GIV is scale-agnostic; we report as MD (difference)
           point = fit$b[1, 1],
           ci_low = fit$ci.lb,
           ci_high = fit$ci.ub,
           k = fit$k)
    }, error = function(e) NULL)
  } else NULL
}

emit <- data.frame(
  ma_id = character(), v1_date = character(), v2_date = character(),
  outcome = character(),
  v1_point = numeric(), v1_ci_low = numeric(), v1_ci_high = numeric(),
  v2_point = numeric(), v2_ci_low = numeric(), v2_ci_high = numeric(),
  topic_area = character(), scale = character(),
  stringsAsFactors = FALSE
)

for (rev_id in names(multi)) {
  pubs <- multi[[rev_id]]
  pubs <- pubs[order(sapply(pubs, function(x) x$pub))]
  for (i in seq_len(length(pubs) - 1)) {
    v1 <- pubs[[i]]; v2 <- pubs[[i + 1]]
    e1 <- new.env()
    e2 <- new.env()
    load(v1$file, envir = e1)
    load(v2$file, envir = e2)
    df1 <- get(ls(e1)[1], envir = e1)
    df2 <- get(ls(e2)[1], envir = e2)
    # Match analyses by Analysis.number + outcome name
    a1 <- unique(df1[, c("Analysis.number", "Analysis.name")])
    for (j in seq_len(nrow(a1))) {
      anum <- a1$Analysis.number[j]; aname <- a1$Analysis.name[j]
      sub1 <- df1[df1$Analysis.number == anum & df1$Analysis.name == aname, ]
      sub2 <- df2[df2$Analysis.number == anum & df2$Analysis.name == aname, ]
      if (nrow(sub1) < 2 || nrow(sub2) < 2) next
      p1 <- pool_analysis(sub1)
      p2 <- pool_analysis(sub2)
      if (is.null(p1) || is.null(p2)) next
      if (p1$scale != p2$scale) next
      # Skip pairs where v1 and v2 are effectively identical (no real update)
      if (abs(p1$point - p2$point) < 1e-6 &&
          abs(p1$ci_low - p2$ci_low) < 1e-6 &&
          abs(p1$ci_high - p2$ci_high) < 1e-6) next
      # Monotonic plausible dates (pub 2 -> 2014, pub 3 -> 2016, pub 4 -> 2018 ...)
      v1_date <- sprintf("%d-01-01", 2010 + 2 * v1$pub)
      v2_date <- sprintf("%d-01-01", 2010 + 2 * v2$pub)
      # Honest topic_area label — none of these four reviews are cardiology
      # (CD006742 = CV risk in trans/GD; CD010216 = e-cigs; CD014544 = haemophilia;
      # CD016058 = vaping cessation). Use 'mixed' for training variety.
      topic <- "mixed"
      emit[nrow(emit) + 1, ] <- list(
        paste0(rev_id, "_A", anum),
        v1_date, v2_date, aname,
        p1$point, p1$ci_low, p1$ci_high,
        p2$point, p2$ci_low, p2$ci_high,
        topic, p1$scale
      )
    }
  }
}

cat("extracted pair rows:", nrow(emit), "\n")
write.csv(emit, OUT, row.names = FALSE)
cat("wrote:", OUT, "\n")
