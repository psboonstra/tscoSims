# Shared policy for a declared outcome level with no observations.
#
# One policy for all four methods: fit over the observed support, tag the
# replicate naming the level, keep it in the denominator, and let the recorded
# `df` say which hypothesis was actually tested.
#
# Without this the three methods failed three different ways on the same
# dataset: `mr` silently dropped from df 5 to df 4 and reported a clean test of
# a different null, `tsco_popo` returned fit_ok = FALSE with zero warnings so
# the replicate left the denominator unannounced, and only `cppo` tagged it.
# Since the 2026-09-17 tsco fix the package itself warns and fits over the
# observed levels, so `tsco_*` now behaves like the others; the tag below
# normalises the wording so all four are greppable the same way.

#' Declared outcome levels with no observations in `dat`.
unobserved_levels <- function(dat, levels_y) {
  counts <- tabulate(match(as.character(dat$Y), levels_y), length(levels_y))
  levels_y[counts == 0L]
}

#' Tag naming the unobserved levels, in the shared format.
#'
#' @param method Short method name, used as the tag prefix.
#' @param missing Character vector of unobserved level labels.
#' @param extra Optional clause appended after a semicolon.
unobserved_level_tag <- function(method, missing, extra = NULL) {
  if (length(missing) == 0L) {
    return(character())
  }

  msg <- sprintf(
    "%s: outcome level(s) %s unobserved; estimates and test are conditional on the observed support",
    method, paste(missing, collapse = ", ")
  )

  if (!is.null(extra)) msg <- paste0(msg, "; ", extra)

  msg
}

#' Tag when a test lost degrees of freedom relative to full outcome support.
#'
#' Derived from the fitted df rather than assumed, so it catches any cause, not
#' only an unobserved level.
df_shortfall_tag <- function(method, df_actual, df_full) {
  if (!is.finite(df_actual) || !is.finite(df_full) || df_actual >= df_full) {
    return(character())
  }

  sprintf(
    "%s: df = %g, not the %g expected under full outcome support -- a different null is being tested",
    method, df_actual, df_full
  )
}
