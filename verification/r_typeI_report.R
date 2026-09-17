suppressMessages({library(dplyr); library(tibble); library(tidyr)})
r <- bind_rows(lapply(list.files("scratch_typeI", full.names = TRUE), readRDS))
cat("replicates:", n_distinct(paste(r$array_id, r$sim_num)), " rows:", nrow(r), "\n")
cat("scenario:", unique(r$scenario), " n:", unique(r$n), " methods:", paste(sort(unique(r$method)), collapse=", "), "\n\n")

N <- n_distinct(paste(r$array_id, r$sim_num))

cat("================ 1. size at nominal 5% ================\n")
s <- r |> group_by(method) |>
  summarize(reps = n(), fit_ok = sum(fit_ok), test_ok = sum(test_ok),
            rej = sum(reject), size = mean(reject),
            size_among_testable = mean(reject[test_ok]),
            se = sqrt(mean(reject) * (1 - mean(reject)) / n()),
            n_warn_reps = sum(n_warnings > 0), .groups = "drop") |>
  mutate(across(c(size, size_among_testable, se), ~round(.x, 4)))
print(as.data.frame(s))
cat("\n  MC SE at a true 0.05 with", N, "reps: ", round(sqrt(.05*.95/N), 4), "\n")

cat("\n================ 2. paired differences against PO ================\n")
w <- r |> select(array_id, sim_num, method, reject, test_ok) |>
  pivot_wider(names_from = method, values_from = c(reject, test_ok))
po <- w$reject_po
for (m in setdiff(sort(unique(r$method)), "po")) {
  x <- w[[paste0("reject_", m)]]
  d <- x - po
  # McNemar-style paired SE on the difference of correlated proportions
  b <- sum(x & !po); c_ <- sum(!x & po)
  se <- sqrt((b + c_) - (b - c_)^2 / N) / N
  z <- if (se > 0) mean(d) / se else NA
  cat(sprintf("  %-10s size %.4f   diff vs po %+.4f   paired SE %.4f   z = %+.2f   (disagree %d/%d)\n",
              m, mean(x), mean(d), se, z, b + c_, N))
}
cat(sprintf("  %-10s size %.4f   (control: correctly specified under the null)\n", "po", mean(po)))

cat("\n================ 3. cppo: reject x warnings ================\n")
cp <- r |> filter(method == "cppo") |>
  mutate(tag = case_when(
    grepl("refit by constrained direct ML", warnings) & grepl("boundary", warnings) ~ "refit + boundary",
    grepl("refit by constrained direct ML", warnings) ~ "refit only",
    grepl("boundary", warnings) ~ "boundary only",
    n_warnings > 0 ~ "other warning",
    TRUE ~ "clean"))
print(cp |> group_by(tag) |>
        summarize(reps = n(), share = round(n()/N, 4), rej = sum(reject),
                  size = round(mean(reject), 4), .groups="drop") |> as.data.frame())
cat("\n  empty exposed-arm cell (category 4):", sum(cp$n_exp_cell == 0),
    sprintf(" = %.4f\n", mean(cp$n_exp_cell == 0)))
cat("  category 4 empty in BOTH arms:      ", sum(cp$n_cell == 0),
    sprintf(" = %.4f\n", mean(cp$n_cell == 0)))
cat("\n  --- tag x empty-cell ---\n")
print(table(tag = cp$tag, empty_exposed = cp$n_exp_cell == 0))

cat("\n================ 4. cppo size with and without the boundary reps ================\n")
onb <- grepl("boundary", cp$warnings)
cat(sprintf("  all reps            %4d   size %.4f\n", nrow(cp), mean(cp$reject)))
cat(sprintf("  boundary reps       %4d   size %.4f\n", sum(onb), mean(cp$reject[onb])))
cat(sprintf("  interior reps       %4d   size %.4f\n", sum(!onb), mean(cp$reject[!onb])))
cat(sprintf("  po, same interior   %4d   size %.4f\n", sum(!onb),
            mean(po[match(paste(cp$array_id, cp$sim_num)[!onb], paste(w$array_id, w$sim_num))])))

cat("\n================ 5. LRT statistic distribution vs chisq_2 ================\n")
print(r |> group_by(method) |>
        summarize(mean_stat = round(mean(stat[test_ok]), 3),
                  median_stat = round(median(stat[test_ok]), 3),
                  q95 = round(quantile(stat[test_ok], .95), 3),
                  max = round(max(stat[test_ok]), 2),
                  df = paste(unique(df[test_ok]), collapse=","), .groups="drop") |> as.data.frame())
cat(sprintf("\n  chisq_2 reference: mean 2, median %.3f, 95%% %.3f\n",
            qchisq(.5,2), qchisq(.95,2)))
cat("  (mr and tsco_popo carry different df; compare each to its own reference)\n")

cat("\n================ 6. size at other nominal levels (cppo vs po) ================\n")
for (a in c(0.10, 0.05, 0.01)) {
  pv <- r |> select(array_id, sim_num, method, p_value) |>
    pivot_wider(names_from = method, values_from = p_value)
  cat(sprintf("  alpha = %.2f   po %.4f   cppo %.4f   mr %.4f   tsco_popo %.4f\n", a,
              mean(pv$po < a, na.rm=TRUE), mean(pv$cppo < a, na.rm=TRUE),
              mean(pv$mr < a, na.rm=TRUE), mean(pv$tsco_popo < a, na.rm=TRUE)))
}
saveRDS(r, "scratch_typeI_all.rds")
