# Provenance of the installed tsco, recorded with every replicate.
#
# WHY THIS EXISTS
# ---------------
# The study's results depend on a package under active development, and
# `packageVersion("tsco")` returns 0.0.0.9000 both before and after the
# 2026-09-17 unobserved-level fix. A version string alone therefore cannot
# answer "which tsco produced this run", which is the one question a results
# file has to be able to answer six months later.
#
# The git commit can answer it, but an installed R package does not carry one:
# after a local `R CMD INSTALL`, `packageDescription("tsco")` has no Remote*
# fields and no hash anywhere. So it has to be stamped in AT INSTALL TIME --
# which is also the only moment the source tree is guaranteed to be present. A
# SLURM node running run_sims.R has the library, not the repository, so asking
# git anything at run time is not an option.
#
# A hash on its own also overstates what it knows, because installs here are
# routinely from a modified working tree. Hence the companion dirty flag,
# captured at the same moment and by the same mechanism.
#
#   install_tsco_stamped()  stamps commit + dirty into a staged DESCRIPTION and
#                           installs from it. The package's own DESCRIPTION is
#                           never modified.
#   tsco_provenance()       reads them back, degrading to NA rather than
#                           failing when tsco was installed some other way.

#' Provenance of the installed package, as a one-row tibble.
#'
#' Picks up `Config/tsco/commit` (written by `install_tsco_stamped()`) and
#' `RemoteSha` (written by `remotes::install_github()`), so either install route
#' is recorded. Returns NA columns rather than erroring if neither is present --
#' a missing hash should not stop a simulation.
tsco_provenance <- function(pkg = "tsco") {

  na_row <- tibble::tibble(
    tsco_version = NA_character_,
    tsco_commit  = NA_character_,
    tsco_dirty   = NA
  )

  d <- tryCatch(utils::packageDescription(pkg), error = function(e) NULL)
  if (is.null(d) || !is.list(d)) {
    return(na_row)
  }

  field <- function(nm) {
    v <- d[[nm]]
    if (is.null(v) || is.na(v) || !nzchar(v)) NA_character_ else as.character(v)
  }

  commit <- field("Config/tsco/commit")
  if (is.na(commit)) commit <- field("RemoteSha")

  dirty <- field("Config/tsco/dirty")

  tibble::tibble(
    tsco_version = as.character(utils::packageVersion(pkg)),
    tsco_commit  = commit,
    # NA, not FALSE, when nothing was stamped: "we did not record it" and "the
    # tree was clean" are different claims and must not be conflated.
    tsco_dirty   = if (is.na(dirty)) NA else identical(tolower(dirty), "true")
  )
}


#' Install a package source tree, stamping its git state into DESCRIPTION.
#'
#' @param pkg_dir Path to the package source (a git working tree).
#' @param lib Library to install into; NULL for the default.
#' @param r_version_floor Optionally rewrite `Depends: R (>= x)` to this
#'   version in the staged copy. The cloud container runs R 4.3.3 against tsco's
#'   declared 4.4.0 floor, and relaxing it in a throwaway copy is how that is
#'   handled without touching the repository.
#' @param ... Passed to `R CMD INSTALL` as extra arguments.
#'
#' Returns the git state that was stamped, invisibly. The installed version is
#' deliberately not included -- call `tsco_provenance()` after a fresh session
#' for that, since the package is not reloaded here.
install_tsco_stamped <- function(pkg_dir,
                                 lib = NULL,
                                 r_version_floor = NULL,
                                 args = "--no-docs") {

  pkg_dir <- normalizePath(pkg_dir, mustWork = TRUE)

  # Quiet, non-throwing git. A source tree that is not a repository, or a
  # machine without git, yields NA rather than an error.
  git <- function(...) {
    out <- suppressWarnings(tryCatch(
      system2("git", c("-C", shQuote(pkg_dir), ...), stdout = TRUE, stderr = FALSE),
      error = function(e) NULL
    ))
    status <- attr(out, "status")
    if (is.null(out) || (!is.null(status) && status != 0L)) character() else out
  }

  commit <- git("rev-parse", "--short", "HEAD")
  commit <- if (length(commit)) commit[1] else NA_character_

  # `git status --porcelain` lists modified AND untracked files; both mean the
  # installed source differs from the named commit, which is what the flag is
  # for.
  dirty <- if (is.na(commit)) NA else length(git("status", "--porcelain")) > 0L

  dest <- file.path(tempfile("tsco_stage_"), basename(pkg_dir))
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  file.copy(pkg_dir, dirname(dest), recursive = TRUE)

  desc_path <- file.path(dest, "DESCRIPTION")
  txt <- readLines(desc_path, warn = FALSE)

  if (!is.null(r_version_floor)) {
    txt <- sub("R \\(>=[^)]*\\)", sprintf("R (>= %s)", r_version_floor), txt)
  }

  # Appending whole new fields at the end is safe; rewriting DESCRIPTION with
  # write.dcf is not, because it reflows continuation lines.
  txt <- txt[!grepl("^Config/tsco/(commit|dirty):", txt)]
  txt <- c(txt,
           sprintf("Config/tsco/commit: %s", if (is.na(commit)) "unknown" else commit),
           sprintf("Config/tsco/dirty: %s", if (is.na(dirty)) "unknown" else tolower(as.character(dirty))))
  writeLines(txt, desc_path)

  cmd_args <- c("CMD", "INSTALL", args,
                if (!is.null(lib)) c("-l", shQuote(lib)),
                shQuote(dest))
  status <- system2(file.path(R.home("bin"), "R"), cmd_args)

  if (status != 0L) {
    stop("R CMD INSTALL failed for ", pkg_dir)
  }

  invisible(tibble::tibble(
    tsco_commit = commit,
    tsco_dirty  = dirty
  ))
}
