source(file.path(dirname(sys.frame(1)$ofile), "_pairwise70_paths.R"))
DATA_DIR <- discover_pairwise70_root()

load(file.path(DATA_DIR, "CD001072_pub2_data.rda"))
x <- get(ls()[1])
cat("columns:", paste(colnames(x), collapse=","), "\n")
cat("n_analyses:", length(unique(x$Analysis.number)), "\n")
cat("year_range:", paste(range(x$Study.year, na.rm=TRUE), collapse="-"), "\n")
cat("years_available:", sum(!is.na(x$Study.year)), "of", nrow(x), "\n")
cat("sample rows analysis 1:\n")
sub <- x[x$Analysis.number == 1, c("Study","Study.year","Experimental.cases","Experimental.N","Control.cases","Control.N","GIV.Mean","GIV.SE")]
print(head(sub, 8))
