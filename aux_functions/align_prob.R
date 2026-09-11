# Puts a matrix of predicted probabilities into the canonical outcome-level
# column order, whatever the fitting backend chose to call its columns.

align_prob <- function(p, levels_y) {
  p <- as.matrix(p)

  if (is.null(colnames(p))) {
    colnames(p) <- levels_y
  }

  missing <- setdiff(levels_y, colnames(p))

  if (length(missing) > 0L) {
    if (ncol(p) == length(levels_y)) {
      colnames(p) <- levels_y
    } else {
      stop("Probability columns do not align.")
    }
  }

  p[, levels_y, drop = FALSE]
}
