# Task 3: what does VGAM actually do on the datasets where the exposed arm has
# no observations in the second-highest category?
#
# For each replicate: is the exposed-arm cell empty; does VGAM's fit stay in the
# CPPO parameter space (cppo_vglm_valid); what LRT does VGAM report; what LRT
# does the constrained direct fit report; and how far is the constrained fit
# from the boundary.
suppressMessages({library(tsco); library(VGAM); library(dplyr); library(tibble)})
options(warn = -1)

levels_y <- as.character(0:5); K <- length(levels_y); M <- K - 1L
source("sim_functions/true_probs.R")
source("sim_functions/score_method.R")
source("aux_functions/safe_fit.R"); source("aux_functions/align_prob.R")
source("aux_functions/vglm_helpers.R")
source("methods/cppo.R")

G <- cppo_G(levels_y)
n <- 200
nrep <- as.integer(Sys.getenv("NREP", "600"))
master_seed <- 20260710

out <- vector("list", nrep)

for (i in seq_len(nrep)) {
  set.seed(master_seed + 100000L + i)
  dat <- make_covariates(n)
  p_true <- true_prob_scenario(dat, "null")
  dat$Y <- draw_ordinal(p_true, levels_y = levels_y)

  n_exp_cell <- sum(dat$A == 1 & dat$Y == as.character(K - 2L))   # category 4, exposed
  n_unexp_cell <- sum(dat$A == 0 & dat$Y == as.character(K - 2L))

  ff <- safe_fit(VGAM::vglm(Y ~ A + W, data = dat,
          family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
          constraints = cppo_constraints(levels_y, G, with_A = TRUE)))
  fr <- safe_fit(VGAM::vglm(Y ~ W, data = dat,
          family = VGAM::cumulative(link = "logitlink", parallel = FALSE, reverse = FALSE),
          constraints = cppo_constraints(levels_y, G, with_A = FALSE)))

  vgam_err   <- !(ff$ok && fr$ok)
  vgam_valid <- !vgam_err && cppo_vglm_valid(ff$fit) && cppo_vglm_valid(fr$fit)

  # VGAM's own numbers, whether or not the fit is admissible.
  v_stat <- v_gamma <- v_minmu <- v_iter <- NA_real_
  if (!vgam_err) {
    tst <- vglm_lrt(ff$fit, fr$fit)
    v_stat <- tst["stat"]
    cf <- coef(ff$fit)
    v_gamma <- if ("deviation" %in% names(cf)) cf[["deviation"]] else NA_real_
    v_minmu <- min(ff$fit@fitted.values)
    v_iter  <- ff$fit@iter
  }

  # Constrained direct ML, always.
  Yidx <- match(as.character(dat$Y), levels_y)
  X <- cbind(A = dat$A, W = dat$W)
  full <- cppo_direct_fit(X, Yidx, a_col = 1L, G = G, K = K)
  red  <- po_direct_fit(X[, "W", drop = FALSE], Yidx, K = K)
  d_stat <- 2 * (full$loglik - red$loglik)

  out[[i]] <- tibble(
    i = i, n_exp_cell = n_exp_cell, n_unexp_cell = n_unexp_cell,
    vgam_err = vgam_err, vgam_valid = vgam_valid,
    vgam_iter = as.numeric(v_iter), vgam_minmu = as.numeric(v_minmu),
    vgam_gamma = as.numeric(v_gamma), vgam_stat = as.numeric(v_stat),
    # Distance to the nearer of the two faces. (Was `full$dist_to_boundary`,
    # a field the fitter no longer returns -- review issue 4.)
    direct_stat = d_stat, dist_bdry = min(full$slack_exposed, full$slack_unexposed),
    n_warn_vgam = length(c(ff$warnings, fr$warnings))
  )
  if (i %% 100 == 0) cat("  ...", i, "\n", file = stderr())
}

r <- bind_rows(out)
saveRDS(r, "scratch_vgam_boundary.rds")

cat("\n================ n =", n, ", nrep =", nrep, ", scenario = null ================\n")
cat(sprintf("empty exposed-arm cell (cat %d):  %d / %d = %.3f\n",
            K - 2L, sum(r$n_exp_cell == 0), nrep, mean(r$n_exp_cell == 0)))
cat(sprintf("VGAM errored:                     %d\n", sum(r$vgam_err)))
cat(sprintf("VGAM fit judged INVALID:          %d / %d = %.4f\n",
            sum(!r$vgam_valid), nrep, mean(!r$vgam_valid)))
cat(sprintf("VGAM hit maxit (30):              %d\n", sum(r$vgam_iter >= 30, na.rm = TRUE)))
cat(sprintf("VGAM raised >=1 warning:          %d\n", sum(r$n_warn_vgam > 0)))

cat("\n--- cross-tab: empty cell  x  VGAM fit valid ---\n")
print(table(empty_cell = r$n_exp_cell == 0, vgam_valid = r$vgam_valid))

cat("\n--- VGAM diagnostics by empty-cell status ---\n")
print(r |> group_by(empty = n_exp_cell == 0) |>
        summarize(n = n(),
                  med_gamma = median(vgam_gamma, na.rm = TRUE),
                  min_gamma = min(vgam_gamma, na.rm = TRUE),
                  max_abs_gamma = max(abs(vgam_gamma), na.rm = TRUE),
                  min_minmu = min(vgam_minmu, na.rm = TRUE),
                  med_iter = median(vgam_iter, na.rm = TRUE),
                  max_iter = max(vgam_iter, na.rm = TRUE),
                  med_vgam_stat = median(vgam_stat, na.rm = TRUE),
                  max_vgam_stat = max(vgam_stat, na.rm = TRUE),
                  med_direct_stat = median(direct_stat, na.rm = TRUE),
                  max_direct_stat = max(direct_stat, na.rm = TRUE)) |>
        as.data.frame())

cat("\n--- on-boundary rate of the CONSTRAINED fit, by empty-cell status ---\n")
print(r |> group_by(empty = n_exp_cell == 0) |>
        summarize(n = n(), on_bdry = mean(dist_bdry < 1e-6),
                  med_dist = median(dist_bdry)) |> as.data.frame())

crit <- qchisq(0.95, 2)
cat("\n--- rejection at nominal 5%, chisq_2 reference ---\n")
print(r |> group_by(empty = n_exp_cell == 0) |>
        summarize(n = n(),
                  rej_vgam = mean(vgam_stat > crit, na.rm = TRUE),
                  rej_direct = mean(direct_stat > crit)) |> as.data.frame())
cat(sprintf("\noverall: VGAM %.4f   direct %.4f\n",
            mean(r$vgam_stat > crit, na.rm = TRUE), mean(r$direct_stat > crit)))

cat("\n--- largest |vgam_stat - direct_stat| ---\n")
print(r |> mutate(gap = abs(vgam_stat - direct_stat)) |> arrange(desc(gap)) |>
        select(i, n_exp_cell, vgam_valid, vgam_iter, vgam_gamma, vgam_minmu,
               vgam_stat, direct_stat, dist_bdry, gap) |> head(12) |> as.data.frame())
