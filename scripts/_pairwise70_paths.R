# sentinel:skip-file  (P0-hardcoded-local-path: dedicated path-discovery helper;
# the candidate drive literals are the canonical env-var-with-fallback pattern.)
# Pairwise70 path discovery — single source of truth for R-side scripts.
#
# Candidate drives are isolated here so the rest of the R codebase doesn't
# contain literal drive letters. Callers go through discover_pairwise70_root()
# which honours PAIRWISE70_ROOT env var before falling back to candidates.
#
# Matches the Python-side _aact_paths.py discovery pattern. Update candidates
# via session memory when the data location moves.

.PAIRWISE70_CANDIDATE_ROOTS <- c(
  "C:/Projects/Pairwise70/data",
  "D:/Pairwise70/data"
)

discover_pairwise70_root <- function(cli_root = NULL) {
  # Resolution order: cli_root > PAIRWISE70_ROOT env > candidate roots.
  # Stops with informative message if nothing resolves.
  if (!is.null(cli_root) && nzchar(cli_root)) {
    return(cli_root)
  }
  env <- Sys.getenv("PAIRWISE70_ROOT", unset = "")
  if (nzchar(env)) {
    return(env)
  }
  for (cand in .PAIRWISE70_CANDIDATE_ROOTS) {
    if (dir.exists(cand)) {
      return(cand)
    }
  }
  stop(
    "Pairwise70 data root not found. Set PAIRWISE70_ROOT env var or pass ",
    "--pairwise70-root CLI arg. Searched: ",
    paste(.PAIRWISE70_CANDIDATE_ROOTS, collapse = ", ")
  )
}
