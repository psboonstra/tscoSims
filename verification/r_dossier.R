# A per-replicate dossier. Set array_id = 1 in run_sims.R and these sim_num
# values reproduce exactly, because the seed is master_seed + array_id*1e5 + i.
#
#   Rscript verification/r_dossier.R                        # null, n = 200
#   IVALS=17,56 Rscript verification/r_dossier.R
#   SCENARIO=cppo_alt N=200 IVALS=7,4 Rscript verification/r_dossier.R
suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)

levels_y <- as.character(0:5); cutoff_level <- "5"; K <- length(levels_y)
master_seed <- 20260710; array_id <- 1L; alpha <- 0.05
n <- as.integer(Sys.getenv("N", "200"))
scenario <- Sys.getenv("SCENARIO", "null")
source("sim_functions/score_method.R"); source("sim_functions/true_probs.R")
source("sim_functions/dataset_diagnostics.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")
source("methods/po.R"); source("methods/mr.R"); source("methods/cppo.R"); source("methods/tsco.R")
G <- cppo_G(levels_y)

set.seed(master_seed + 999); test_dat <- make_covariates(200)
p_true_test <- true_prob_scenario(test_dat, scenario)

ivals <- as.integer(strsplit(Sys.getenv("IVALS", "1,12,17,56,82,104,123,133"), ",")[[1]])

for (i in ivals) {
  seed_i <- master_seed + array_id * 100000L + i
  set.seed(seed_i)
  dat <- make_covariates(n)
  dat$Y <- draw_ordinal(true_prob_scenario(dat, scenario), levels_y = levels_y)

  cat(sprintf("\n=========== %s, n = %d, i = %d   (data_seed = %d) ===========\n", scenario, n, i, seed_i))
  tb <- table(A = dat$A, Y = factor(as.character(dat$Y), levels = levels_y))
  cat("  counts of Y by arm (categories 0..5):\n")
  cat(sprintf("    A = 0:  %s\n", paste(sprintf("%3d", tb["0", ]), collapse = " ")))
  cat(sprintf("    A = 1:  %s\n", paste(sprintf("%3d", tb["1", ]), collapse = " ")))
  cat(sprintf("  category 4:  exposed %d, unexposed %d\n", tb["1","4"], tb["0","4"]))
  dg <- dataset_diagnostics(dat, levels_y)
  cat(sprintf("  diagnostics: n_empty_levels %d  n_empty_exp %d  n_empty_unexp %d  min_cell %d\n",
      dg$n_empty_levels, dg$n_empty_exp, dg$n_empty_unexp, dg$min_cell))

  ff <- safe_fit(VGAM::vglm(Y ~ A + W, data = dat,
          family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
          constraints = cppo_constraints(levels_y, G, with_A = TRUE)))
  fr <- safe_fit(VGAM::vglm(Y ~ W, data = dat,
          family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
          constraints = cppo_constraints(levels_y, G, with_A = FALSE)))

  cat("  VGAM:\n")
  if (!ff$ok || !fr$ok) {
    cat(sprintf("    ERROR in %s model: %s\n",
        paste(c(if(!ff$ok) "full", if(!fr$ok) "reduced"), collapse = "+"),
        c(if(!ff$ok) ff$fit$error, if(!fr$ok) fr$fit$error)[1]))
  } else {
    cat(sprintf("    fit ok; iter %d (maxit %d); min fitted prob %.3e\n",
        ff$fit@iter, ff$fit@control$maxit, min(ff$fit@fitted.values)))
    cat(sprintf("    cppo_vglm_valid: full %s, reduced %s\n",
        cppo_vglm_valid(ff$fit), cppo_vglm_valid(fr$fit)))
    ll <- vglm_loglik(ff$fit)
    cat(sprintf("    loglik full: %s\n", if (is.na(ll)) "NA  <- NaN criterion, LRT unusable" else sprintf("%.4f", ll)))
    t_v <- vglm_lrt(ff$fit, fr$fit)
    cat(sprintf("    VGAM LRT: stat %s  df %g\n",
        if (is.na(t_v["stat"])) "NA" else sprintf("%.4f", t_v["stat"]), t_v["df"]))
  }
  wn <- unique(c(ff$warnings, fr$warnings))
  if (length(wn)) {
    cat("    VGAM warnings:\n")
    for (w in wn) cat(sprintf("      - %s\n", substr(w, 1, 100)))
  } else cat("    VGAM warnings: none\n")

  Yidx <- match(as.character(dat$Y), levels_y)
  X <- cbind(A = dat$A, W = dat$W)
  fl <- cppo_direct_fit(X, Yidx, a_col = 1L, G = G, K = K)
  rd <- po_direct_fit(X[, "W", drop = FALSE], Yidx, K = K)
  # Both faces of the admissible region, on the log-odds scale. The exposed
  # face (gamma >= alpha_{K-1} - alpha_{K-2}) binds when the exposed cell is
  # empty; the cutpoint-spacing face binds when the UNexposed cell is empty.
  faces <- c(exposed = fl$slack_exposed, unexposed = fl$slack_unexposed)
  cat(sprintf("  constrained direct ML: LRT %.4f   gamma %.4f\n",
              2*(fl$loglik - rd$loglik), fl$gamma))
  cat(sprintf("    slack, exposed face (gamma)      %.3e%s\n", faces[["exposed"]],
              if (faces[["exposed"]] < 1e-6) "   <- ON the boundary" else ""))
  cat(sprintf("    slack, cutpoint-spacing face     %.3e%s\n", faces[["unexposed"]],
              if (faces[["unexposed"]] < 1e-6) "   <- ON the boundary" else ""))

  cat("  what fxn_cppo() returns:\n")
  res <- fxn_cppo(dat, test_dat, levels_y)
  cat(sprintf("    fit_ok %s   stat %.4f   df %g   p %.4f   reject %s\n",
      res$fit_ok, res$stat, res$df, res$p_value, res$p_value < alpha))
  if (length(res$warnings)) for (w in res$warnings) cat(sprintf("    tag: %s\n", substr(w, 1, 100)))

  cat("  all four methods:\n")
  fits <- list(po = fxn_po(dat, test_dat, levels_y),
               tsco_popo = fxn_tsco_popo(dat, test_dat, levels_y, cutoff_level),
               mr = fxn_mr(dat, test_dat, levels_y),
               cppo = res)
  for (m in names(fits)) {
    s <- score_method(fits[[m]], p_true_test, alpha)
    cat(sprintf("    %-10s fit_ok %-5s test_ok %-5s df %-4s stat %-8s p %-8s reject %s\n",
        m, s$fit_ok, s$test_ok,
        if (is.na(s$df)) "NA" else format(s$df),
        if (is.na(s$stat)) "NA" else sprintf("%.3f", s$stat),
        if (is.na(s$p_value)) "NA" else sprintf("%.4f", s$p_value), s$reject))
  }
}
