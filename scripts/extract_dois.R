source(file.path(dirname(sys.frame(1)$ofile), "_pairwise70_paths.R"))
DATA_DIR <- discover_pairwise70_root()
OUT_DIR <- file.path(dirname(dirname(sys.frame(1)$ofile)), "cache")

files <- list.files(DATA_DIR,
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
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
writeLines(doi_list, file.path(OUT_DIR, "review_dois.txt"))
cat("unique DOIs:", length(doi_list), "\n")
print(head(doi_list, 5))
