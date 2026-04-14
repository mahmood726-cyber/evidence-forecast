files <- list.files("C:/Projects/Pairwise70/data", pattern = "\\.rda$", full.names = TRUE)
urls <- c()
for (f in files[1:10]) {
  env <- new.env()
  load(f, envir = env)
  x <- get(ls(env)[1], envir = env)
  if ("review.url" %in% colnames(x)) {
    urls <- c(urls, unique(x$review.url)[1])
  } else if ("review_url" %in% colnames(x)) {
    urls <- c(urls, unique(x$review_url)[1])
  }
}
print(urls)
