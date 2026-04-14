files <- list.files("C:/Projects/Pairwise70/data",
                    pattern = "^CD[0-9]+_pub[0-9]+_data\\.rda$", full.names = TRUE)
doi_list <- character(0)
for (f in files) {
  env <- new.env()
  load(f, envir = env)
  x <- get(ls(env)[1], envir = env)
  col <- intersect(c("review.doi", "review_doi"), colnames(x))[1]
  if (!is.na(col)) {
    val <- unique(as.character(x[[col]]))[1]
    if (!is.na(val) && nzchar(val)) doi_list <- c(doi_list, val)
  }
}
doi_list <- unique(doi_list)
dir.create("C:/Models/EvidenceForecast/cache", recursive = TRUE, showWarnings = FALSE)
writeLines(doi_list, "C:/Models/EvidenceForecast/cache/review_dois.txt")
cat("unique DOIs:", length(doi_list), "\n")
print(head(doi_list, 5))
