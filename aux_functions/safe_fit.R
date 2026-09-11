# Evaluates a fitting call, capturing warnings rather than letting them print,
# and converting an error into a value that can be tested with `$ok`.

safe_fit <- function(expr) {
  warnings <- character()

  value <- withCallingHandlers(
    tryCatch(
      expr,
      error = function(e) {
        structure(
          list(error = conditionMessage(e)),
          class = "fit_error"
        )
      }
    ),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  list(
    fit = value,
    warnings = warnings,
    ok = !inherits(value, "fit_error")
  )
}
