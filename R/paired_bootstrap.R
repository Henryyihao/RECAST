#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^", flag, "="), "", hit[[1]]) else default
}
data_dir <- get_arg("--data-dir", "work/server_exports")
out_dir <- get_arg("--out-dir", "results/analysis")
source_dir <- get_arg("--source-dir", "results/source_data")
B <- as.integer(get_arg("--B", "2000"))
seed <- as.integer(get_arg("--seed", "20260930"))
blocks <- as.integer(strsplit(get_arg("--blocks", "2,4,8"), ",", fixed = TRUE)[[1]])
if (!is.finite(B) || B < 100L) stop("--B must be at least 100")
if (!length(blocks) || any(!is.finite(blocks)) || any(blocks < 1L)) stop("Invalid --blocks")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)
read_input <- function(name) {
  path <- file.path(data_dir, name)
  if (!file.exists(path)) stop("Missing ", path)
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}
num <- function(x) suppressWarnings(as.numeric(as.character(x)))
close_enough <- function(a, b) is.finite(a) & is.finite(b) & abs(a - b) <= 1e-8 + 1e-5 * pmax(abs(a), abs(b), 1)

main <- read_input("main_metrics.csv")
bayes_path <- file.path(data_dir, "bayesian_metrics.csv")
bayes <- if (file.exists(bayes_path)) read_input("bayesian_metrics.csv") else main[FALSE, ]
reference_value <- function(domain, method, task, metric) {
  table <- if (grepl("^bayes", method)) bayes else main
  hit <- table[table$domain == domain & table$method == method & table$task == task, , drop = FALSE]
  if (nrow(hit) != 1L || !metric %in% names(hit)) return(NA_real_)
  num(hit[[metric]][1])
}
aliases <- c(recast = "v12_final", light = "abl2_light", zsviews = "chronos2_zsviews_psn",
             naive = "plain_naive", oracle = "plain_oracle")
checks <- read_input("score_recompute_checks.csv")
latest_checks <- read_input("latest_report_recompute_checks.csv")
for (name in setdiff(names(checks), names(latest_checks))) latest_checks[[name]] <- NA
checks <- rbind(checks, latest_checks[, names(checks)])
checks$method_main <- vapply(checks$method, function(m) if (m %in% names(aliases)) aliases[[m]] else m, character(1))
checks$reference_value <- mapply(reference_value, checks$domain, checks$method_main, checks$task, checks$metric)
checks$stored_ok <- !is.finite(num(checks$stored)) | close_enough(num(checks$recomputed), num(checks$stored))
checks$reference_ok <- close_enough(num(checks$recomputed), checks$reference_value)
checks$reference_difference <- num(checks$recomputed) - checks$reference_value
checks$ok <- checks$stored_ok & checks$reference_ok
write.csv(checks, file.path(out_dir, "paired_bootstrap_recompute_check.csv"), row.names = FALSE)
if (any(!checks$ok)) stop("Score check failed; see paired_bootstrap_recompute_check.csv")

forecast_path <- file.path(data_dir, "paired_forecast_losses.csv")
forecast <- if (file.exists(forecast_path)) read_input("paired_forecast_losses.csv") else read_input("paired_forecast_bootstrap_input.csv")
nowcast <- read_input("paired_nowcast_bootstrap_input.csv")
specs <- list(
  forecast = list(data = forecast, metric = "CRPS_s", suffix = "crps_sum",
                  methods = c("recast", "naive", "twostage"),
                  comparisons = c(recast_vs_naive = "naive", recast_vs_cl_pipeline = "twostage")),
  nowcast = list(data = nowcast, metric = "MASE", suffix = "mase_sum",
                 methods = c("recast", "latest_report", "cl"),
                 comparisons = c(recast_vs_latest_report = "latest_report", recast_vs_cl = "cl")))

for (task in names(specs)) {
  spec <- specs[[task]]
  columns <- c("domain", "sid", "origin", "origin_date", "scale", "above_1pct_scale",
               as.vector(rbind(paste0(spec$methods, "_", spec$suffix), paste0(spec$methods, "_count"))))
  if (!all(columns %in% names(spec$data))) stop("Missing ", task, " columns: ", paste(setdiff(columns, names(spec$data)), collapse = ", "))
  specs[[task]]$data <- spec$data[, columns]
  write.csv(specs[[task]]$data, file.path(source_dir, paste0("paired_", task, "_bootstrap_input.csv")), row.names = FALSE)
}
for (name in c("main_metrics.csv", "bayesian_metrics.csv", "score_recompute_checks.csv", "latest_report_recompute_checks.csv")) {
  file.copy(file.path(data_dir, name), file.path(source_dir, name), overwrite = TRUE)
}
rm(forecast, nowcast)
invisible(gc())

prepare_panel <- function(data, spec) {
  if (anyDuplicated(paste(data$sid, data$origin, sep = "::"))) stop("Duplicate series-origin keys")
  series <- sort(unique(as.character(data$sid)))
  origins <- sort(unique(num(data$origin)))
  if (any(!is.finite(origins))) stop("Origin indices must be numeric")
  ii <- match(as.character(data$sid), series); jj <- match(num(data$origin), origins)
  mats <- lapply(spec$methods, function(method) {
    sm <- matrix(NA_real_, length(series), length(origins)); cm <- sm
    sm[cbind(ii, jj)] <- num(data[[paste0(method, "_", spec$suffix)]])
    cm[cbind(ii, jj)] <- num(data[[paste0(method, "_count")]])
    list(sum = sm, count = cm)
  })
  names(mats) <- spec$methods
  list(mats = mats, n_series = length(series), n_origins = length(origins),
       n_series_origins = nrow(data), origin_index_gap_median = median(diff(origins)),
       origin_index_gap_min = min(diff(origins)), origin_index_gap_max = max(diff(origins)))
}
draw_origins <- function(n, block) {
  block <- min(n, block)
  starts <- sample.int(n, ceiling(n / block), replace = TRUE)
  indices <- ((rep(starts, each = block) - 1L + rep(0:(block - 1L), length(starts))) %% n) + 1L
  indices[seq_len(n)]
}
paired_score <- function(a, b, si = seq_len(nrow(a$sum)), oi = seq_len(ncol(a$sum))) {
  av <- a$sum[si, oi, drop = FALSE]; ac <- a$count[si, oi, drop = FALSE]
  bv <- b$sum[si, oi, drop = FALSE]; bc <- b$count[si, oi, drop = FALSE]
  ok_a <- is.finite(av) & is.finite(ac) & ac > 0
  ok_b <- is.finite(bv) & is.finite(bc) & bc > 0
  if (!any(ok_a) || !any(ok_b)) return(c(difference = NA_real_, relative = NA_real_))
  score_a <- sum(av[ok_a]) / sum(ac[ok_a]); score_b <- sum(bv[ok_b]) / sum(bc[ok_b])
  c(difference = score_a - score_b, relative = 100 * (score_a / score_b - 1))
}

results <- list(); metadata <- list(); result_i <- 0L; meta_i <- 0L
for (task in names(specs)) {
  spec <- specs[[task]]
  domains <- sort(unique(spec$data$domain))
  panels <- lapply(domains, function(domain) prepare_panel(spec$data[spec$data$domain == domain, ], spec))
  names(panels) <- domains
  for (domain in domains) {
    panel <- panels[[domain]]
    meta_i <- meta_i + 1L
    metadata[[meta_i]] <- data.frame(task = task, domain = domain, n_series = panel$n_series,
      n_origins = panel$n_origins, n_series_origins = panel$n_series_origins,
      origin_index_gap_median = panel$origin_index_gap_median,
      origin_index_gap_min = panel$origin_index_gap_min, origin_index_gap_max = panel$origin_index_gap_max)
    for (method in spec$methods) {
      mat <- panel$mats[[method]]
      observed <- is.finite(mat$sum) & is.finite(mat$count) & mat$count > 0
      point <- sum(mat$sum[observed]) / sum(mat$count[observed])
      method_main <- if (method %in% names(aliases)) aliases[[method]] else method
      expected <- reference_value(domain, method_main, task, spec$metric)
      if (!close_enough(point, expected)) stop("Input loss does not reproduce ", domain, "/", method, "/", task)
    }
  }
  for (block in blocks) {
    draws <- list(); points <- list()
    for (domain in domains) {
      panel <- panels[[domain]]
      set.seed(seed + 100000L * block + 1000L * match(domain, domains) + match(task, names(specs)))
      reps <- array(NA_real_, c(B, 2L, length(spec$comparisons)),
                    dimnames = list(NULL, c("difference", "relative"), names(spec$comparisons)))
      point <- sapply(spec$comparisons, function(b) paired_score(panel$mats$recast, panel$mats[[b]]))
      for (r in seq_len(B)) {
        si <- sample.int(panel$n_series, panel$n_series, replace = TRUE)
        oi <- draw_origins(panel$n_origins, block)
        for (j in seq_along(spec$comparisons)) {
          reps[r, , j] <- paired_score(panel$mats$recast, panel$mats[[spec$comparisons[[j]]]], si, oi)
        }
      }
      draws[[domain]] <- reps; points[[domain]] <- point
      for (j in seq_along(spec$comparisons)) {
        d <- reps[, 1L, j]; rel <- reps[, 2L, j]
        dci <- quantile(d, c(.025, .975), na.rm = TRUE, names = FALSE)
        rci <- quantile(rel, c(.025, .975), na.rm = TRUE, names = FALSE)
        result_i <- result_i + 1L
        results[[result_i]] <- data.frame(task = task, metric = spec$metric, scope = "domain", domain = domain,
          comparison = names(spec$comparisons)[j], block_length_origin_steps = block,
          n_series = panel$n_series, n_origins = panel$n_origins, n_series_origins = panel$n_series_origins,
          n_cells = sum(panel$mats$recast$count, na.rm = TRUE), point_difference = point[1L, j],
          point_relative_change_pct = point[2L, j], difference_ci_low = dci[1L], difference_ci_high = dci[2L],
          relative_change_ci_low_pct = rci[1L], relative_change_ci_high_pct = rci[2L],
          n_replicates = sum(is.finite(rel)))
      }
      message(task, "/", domain, ": block=", block, " completed")
    }
    for (j in seq_along(spec$comparisons)) {
      d <- rowMeans(sapply(draws, function(x) x[, 1L, j]), na.rm = TRUE)
      rel <- rowMeans(sapply(draws, function(x) x[, 2L, j]), na.rm = TRUE)
      dci <- quantile(d, c(.025, .975), na.rm = TRUE, names = FALSE)
      rci <- quantile(rel, c(.025, .975), na.rm = TRUE, names = FALSE)
      result_i <- result_i + 1L
      results[[result_i]] <- data.frame(task = task, metric = spec$metric, scope = "all_domains_equal_weight", domain = "ALL",
        comparison = names(spec$comparisons)[j], block_length_origin_steps = block,
        n_series = sum(vapply(panels, `[[`, numeric(1), "n_series")),
        n_origins = sum(vapply(panels, `[[`, numeric(1), "n_origins")),
        n_series_origins = sum(vapply(panels, `[[`, numeric(1), "n_series_origins")),
        n_cells = sum(vapply(panels, function(p) sum(p$mats$recast$count, na.rm = TRUE), numeric(1))),
        point_difference = mean(vapply(points, function(x) x[1L, j], numeric(1))),
        point_relative_change_pct = mean(vapply(points, function(x) x[2L, j], numeric(1))),
        difference_ci_low = dci[1L], difference_ci_high = dci[2L],
        relative_change_ci_low_pct = rci[1L], relative_change_ci_high_pct = rci[2L], n_replicates = sum(is.finite(rel)))
    }
  }
}
results <- do.call(rbind, results)
write.csv(results, file.path(out_dir, "paired_bootstrap_ci.csv"), row.names = FALSE)
write.csv(do.call(rbind, metadata), file.path(out_dir, "paired_bootstrap_panel_metadata.csv"), row.names = FALSE)
write.csv(results[results$scope == "all_domains_equal_weight", ], file.path(out_dir, "paired_bootstrap_domain_summary.csv"), row.names = FALSE)
writeLines(c("Bootstrap completed.", paste0("Replicates: ", B), paste0("Seed: ", seed),
  paste0("Circular time-block lengths in archived origin steps: ", paste(blocks, collapse = ", ")),
  "Within each domain, sample series with replacement and time blocks on the shared origin axis; use the full crossed sampled panel.",
  "All methods share sampled series and origins. Scores retain the archived valid-cell weighting.",
  "Equal-domain summaries treat the 11 domains as a fixed benchmark panel.",
  "95% percentile intervals quantify evaluation-panel dependence for a single fitted run; training randomness is not resampled.",
  "Negative differences and relative changes favor RECAST."), file.path(out_dir, "paired_bootstrap_status.txt"))
message("Wrote ", file.path(out_dir, "paired_bootstrap_ci.csv"))
