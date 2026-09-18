# Puts a matrix of predicted probabilities into the canonical outcome-level
# column order, inserting a zero column for any declared level the backend
# did not return.
#
# WHY ZERO, AND WHY BY LABEL (rewritten 2026-09-18, after external review)
# ------------------------------------------------------------------------
# VGAM drops an outcome level with no observations and returns K - 1 columns.
# The previous version of this function stopped there, so on a dataset with a
# globally empty level `po` and `mr` returned p_hat = NA and silently left the
# PREDICTION denominator while `tsco`, which restores the level itself, stayed
# in. That gave the accuracy comparison method-specific denominators on
# exactly the sparse draws the study is about.
#
# The shared policy (aux_functions/unobserved_levels.R) is: fit over the
# observed support and carry a declared-but-unobserved level at zero mass. Zero
# is not a convention here, it is the maximum-likelihood fitted probability of
# a level that was never observed, in every model class in this study. So the
# missing column is inserted as zero and the replicate stays in.
#
# Columns are matched BY LABEL. Every backend used here labels its columns with
# the outcome levels (checked: VGAM cumulative and multinomial, tsco, and the
# direct CPPO fitter sets them explicitly), so an unlabelled matrix indicates a
# calling error and is refused rather than relabelled positionally -- the old
# `if (ncol(p) == length(levels_y)) colnames(p) <- levels_y` could silently
# misassign categories.

align_prob <- function(p, levels_y) {
  p <- as.matrix(p)

  if (is.null(colnames(p))) {
    stop("align_prob: probability matrix has no column labels; cannot align by level.")
  }

  extra <- setdiff(colnames(p), levels_y)
  if (length(extra) > 0L) {
    stop("align_prob: unexpected probability columns: ", paste(extra, collapse = ", "))
  }

  missing <- setdiff(levels_y, colnames(p))
  if (length(missing) > 0L) {
    zeros <- matrix(0, nrow = nrow(p), ncol = length(missing),
                    dimnames = list(NULL, missing))
    p <- cbind(p, zeros)
  }

  p[, levels_y, drop = FALSE]
}

# Validates a probability matrix against the truth it will be scored against.
# Used by score_method() to decide `pred_ok`: a matrix that is the wrong shape,
# non-finite, negative beyond rounding, mislabelled, or whose rows do not sum
# to one is not a prediction and must not enter the accuracy metrics as one.
valid_prob_matrix <- function(p, p_true, tol = 1e-8) {
  if (!is.matrix(p)) return(FALSE)
  if (!identical(dim(p), dim(as.matrix(p_true)))) return(FALSE)
  if (!all(is.finite(p))) return(FALSE)
  if (any(p < -tol)) return(FALSE)
  if (!all(abs(rowSums(p) - 1) <= 1e-6)) return(FALSE)
  cn_true <- colnames(p_true)
  if (!is.null(cn_true) && !identical(colnames(p), cn_true)) return(FALSE)
  TRUE
}
