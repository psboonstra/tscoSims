# Task 2: confirm the S4 slots cppo_vglm_valid() relies on actually exist, and
# that the direct-fit path returns what score_method() expects.
suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble); library(glue)})
options(warn = 1)

levels_y <- as.character(0:5); cutoff_level <- "5"; K <- length(levels_y)
source("sim_functions/true_probs.R")
source("sim_functions/score_method.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")
source("methods/cppo.R")

set.seed(20260710 + 100001)
n <- 200
dat <- make_covariates(n)
p_true <- true_prob_scenario(dat, "null")
dat$Y <- draw_ordinal(p_true, levels_y = levels_y)
test_dat <- make_covariates(2000)
p_true_test <- true_prob_scenario(test_dat, "null")

cat("\n=== A. does the VGAM fit have the slots cppo_vglm_valid() asks for? ===\n")
G <- cppo_G(levels_y)
cat("G =", G, "\n")
fit <- VGAM::vglm(Y ~ A + W, data = dat,
                  family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
                  constraints = cppo_constraints(levels_y, G, with_A = TRUE))

chk <- function(label, expr) {
  v <- tryCatch(eval(expr), error = function(e) structure(conditionMessage(e), class = "err"))
  if (inherits(v, "err")) {
    cat(sprintf("  [FAIL] %-24s error: %s\n", label, v))
  } else {
    cat(sprintf("  [ok  ] %-24s %s\n", label,
                paste0(class(v)[1], " ", paste(dim(v) %||% length(v), collapse = "x"),
                       "  ", paste(utils::head(as.numeric(v), 3), collapse = " "))))
  }
  invisible(v)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

mu   <- chk("fit@fitted.values", quote(fit@fitted.values))
it   <- chk("fit@iter",          quote(fit@iter))
mx   <- chk("fit@control$maxit", quote(fit@control$maxit))
crit <- chk("fit@criterion$loglik", quote(as.numeric(fit@criterion[["loglikelihood"]])))
cat("  slotNames present:", paste(intersect(c("fitted.values","iter","control","criterion"),
                                            slotNames(fit)), collapse = ", "), "\n")
cat("  ncol(fitted) =", ncol(mu), " colnames:", paste(colnames(mu), collapse = ","), "\n")
cat("  cppo_vglm_valid(fit) =", cppo_vglm_valid(fit), "\n")

cat("\n=== B. does the valid-check actually discriminate? ===\n")
# Feed it a deliberately broken object: negative fitted value.
bad <- fit; bad@fitted.values[1, 5] <- -0.01
cat("  cppo_vglm_valid(negative fitted) =", cppo_vglm_valid(bad), "  (want FALSE)\n")
bad2 <- fit; bad2@iter <- bad2@control$maxit
cat("  cppo_vglm_valid(iter == maxit)   =", cppo_vglm_valid(bad2), "  (want FALSE)\n")

cat("\n=== C. vgam path: return shape ===\n")
res_v <- fxn_cppo(dat, test_dat, levels_y, engine = "vgam")
str(res_v[c("fit_ok","stat","df","p_value","warnings")])
cat("\n=== D. direct path: return shape ===\n")
res_d <- fxn_cppo(dat, test_dat, levels_y, engine = "direct")
str(res_d[c("fit_ok","stat","df","p_value","warnings")])

cat("\n=== E. score_method() accepts both ===\n")
for (nm in c("vgam","direct")) {
  r <- if (nm == "vgam") res_v else res_d
  s <- tryCatch(score_method(r, p_true_test, 0.05), error = function(e) conditionMessage(e))
  if (is.character(s)) cat(sprintf("  [FAIL] %s: %s\n", nm, s)) else {
    cat(sprintf("  [ok  ] %-7s stat = %.4f  df = %g  p = %.4f  reject = %s  rps = %.3e\n",
                nm, s$stat, s$df, s$p_value, s$reject, s$rps))
  }
}

cat("\n=== F. do the two engines agree when the constraint is slack? ===\n")
cat(sprintf("  vgam   stat = %.8f   df = %g\n", res_v$stat, res_v$df))
cat(sprintf("  direct stat = %.8f   df = %g\n", res_d$stat, res_d$df))
cat(sprintf("  |diff| = %.2e\n", abs(as.numeric(res_v$stat) - as.numeric(res_d$stat))))
cat("  min fitted Pr(Y=4 | A=1) =", min(fit@fitted.values[dat$A == 1, K - 1L]), "\n")
cat("  exposed-arm count in category 4:", sum(dat$A == 1 & dat$Y == "4"), "\n")
