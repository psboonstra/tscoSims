# Did the backend converge? `safe_fit()` only distinguishes a thrown error from
# a returned object, and every backend here will happily return an object
# whose optimiser gave up (review issue 12). This asks each one the question
# it can answer:
#
#   rms::orm     `$fail`: TRUE when the Newton iterations failed.
#   VGAM::vglm   `@iter < @control$maxit`: IRLS stopped before the ceiling.
#                Also requires a finite log-likelihood, because a fit that
#                overshoots reports NaN there.
#   tsco         both stage fits, whichever backend each used.
#
# Returns TRUE / FALSE, never NA: an object of unknown class is treated as
# converged so that this check can only ever REMOVE replicates for a reason it
# can name, not for one it cannot.

fit_converged <- function(fit) {
  if (inherits(fit, "tsco")) {
    return(fit_converged(fit$fit_stage1) && fit_converged(fit$fit_stage2))
  }

  if (inherits(fit, "orm")) {
    return(!isTRUE(fit$fail))
  }

  if (inherits(fit, "vglm")) {
    iter_ok <- tryCatch(
      is.null(fit@control$maxit) || fit@iter < fit@control$maxit,
      error = function(e) TRUE
    )
    ll <- tryCatch(as.numeric(fit@criterion["loglikelihood"]), error = function(e) NA_real_)
    return(isTRUE(iter_ok) && length(ll) == 1L && is.finite(ll))
  }

  TRUE
}
