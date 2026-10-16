# Decisions

A dated log of the choices made in this project and the reasons for them, including bug fixes. All fixed settings live in `src/pvcal/config.py`.

## 2026-10-01

**Linear quantile regression uses statsmodels `QuantReg`.**
It is typically faster than scikit-learn's `QuantileRegressor` on a training set of this size (about 30,000 daylight hours), which matters because the model is fitted once per quantile level per task. statsmodels is part of the `models` extra in `pyproject.toml`.

## 2026-10-05

**Data source and verification.**
We use `GEFCom2014.zip` (126,360,077 bytes) from the competition organizer's link (see `data/README.md`) and keep only `GEFCom2014-S_V2.zip/Solar`, in `data/raw/Solar`, which is not committed. All real-data tests pass, so every data fact the code relies on holds.

**The baseline results reproduce exactly.**
Rerunning the baselines gives 0.0376 (official benchmark) and 0.0228 (persistence ensemble), and the per-task scores file is identical to the earlier one. The two baseline figures were regenerated; their bytes differ only because the installed matplotlib renders slightly differently.

**Bug fix: summaries are written as UTF-8.**
`Path.write_text()` without `encoding=` uses the system default, which is cp1252 on Windows, so the "–" placeholders in the baseline summary were saved as an invalid byte. Every script now passes `encoding="utf-8"` when it writes text, and reconfigures its console output to UTF-8.

**Settings live in `src/pvcal/config.py`.**
Every fixed number (seed, development tasks, tuning grid, interval levels, calibration window, ACI step sizes, bootstrap size, cloud-cover cut-off date) is a named constant there with a one-line reason, so no setting is hidden inside a function.

**`finalize()` caps at 1 and can count what it fixes.**
Power is a fraction of capacity, so quantiles are clipped to [0, 1] (previously only at 0). An optional `log` dict counts, on daylight rows and before the clean-up, the rows whose quantiles crossed and the values above 1. Rerunning the baselines left their scores unchanged: the persistence ensemble's highest quantile over the test months is 0.990761, below the cap, even though observed power reaches 1.00355 once (zone 3, 2013-11-04 02:00 UTC).

**The linear model's indicator columns are built by explicit comparisons.**
The hour and zone columns are built as `(hour == h)` for each training hour except the reference one. An hour never seen in training then gets all zeros, which means "treat it like the reference hour". Building a `pd.Categorical` from values outside its categories raises a deprecation warning in pandas 3.0 and will become an error, and the explicit version is easier to read.

**Both quantile models fit all 99 levels; the 21-level fallback is not used.**
We set a runtime budget in advance (`FALLBACK_IF_HOURS` = 3 hours for the Track 1 run) and timed task 15, the largest training set (29,558 daylight rows): all 99 linear fits took 143 s and all 99 LightGBM fits with 800 trees took 164 s. The worst case for Track 1 (16 task-runs per model, including the leakage reruns) is about 82 minutes, well under the budget, so fitting 21 levels and interpolating was not needed. The slowest single fit is the extreme level tau = 0.01 (15.8 s against about 1.5 s for the median).

**LightGBM settings frozen: num_leaves = 31, min_child_samples = 100, n_estimators = 300.**
Chosen on development tasks 1-3 only (`results/tables/tuning_lightgbm.csv`), by mean daytime pinball loss at levels 0.05, 0.25, 0.5, 0.75 and 0.95. It scored 0.0220594, just below the same leaves and leaf size with 800 trees (0.0220605). Scores equal to five decimal places count as a tie, which goes to fewer trees; that rule gives the same choice here. `random_state=0` is LightGBM's `seed`, with `deterministic=True` and `force_row_wise=True`, so reruns give identical numbers.

**Leakage checks inside the run scripts.**
The baselines are checked on every test month (cheap). The two quantile models are checked on tasks 4 and 15: each check reruns all 99 fits twice, and it tests the code path rather than the data, so two tasks are enough. The Track 2 script checks the split model on tasks 4 and 15 and ACI's day-by-day causality on the real sequence.

**Every table is written twice, and the README is filled from the tables.**
`pvcal.tables.save_table` writes a full-precision CSV and a Markdown copy. `scripts/report.py` copies the Markdown into README.md between `<!-- BEGIN x -->` and `<!-- END x -->` markers and builds the short answer and key numbers from the CSVs, so no result in the README is typed by hand. The verdicts in the short answer ("falls short of its promise", "is on target", "covers more than it promises") are chosen by whether each 95% confidence interval contains the promised level, and the Winkler comparison by the scores themselves. Protocol constants (3 zones, 99 levels, 60 days, UTC+10) and data facts checked by the tests are written in the README text.

**Track 2 setup.**
V2 trains LightGBM (the frozen settings) on daylight rows up to 00:00 UTC 60 days before the test month, at levels 0.05, 0.10, 0.90 and 0.95; it predicts the 60-day calibration window and the test month, sorted and clipped to [0, 1]. V3 (CQR) pools the three zones' daylight calibration rows (`results/tables/cqr_adjustments.csv` lists each month's adjustment). V4 (ACI) uses the same calibration scores, refreshed at each task; it starts at the nominal miss rate at task 1 and carries its level through task 15. Yearly coverage pools all daylight hours of the test months (each hour counts equally), unlike Track 1's equal weight per month.

**CQR edge case: an interval narrowed past zero width collapses to its midpoint.**
A negative adjustment larger than half an interval's width would put the lower end above the upper end, and the CQR rule does not specify this case. A zero-width interval at the midpoint is the least surprising choice. It never happened in the real runs (`results/tables/track2_diagnostics.csv`, column `cqr_collapsed_rows`), and neither did any ACI edge case (alpha <= 0, alpha >= 1, rank > n).

**ACI step size gamma = 0.005, chosen on development tasks 1-3.**
From `results/tables/aci_gamma_selection.csv`, 90% coverage on tasks 1-3 was 0.899254 (gamma 0.005), 0.89654 (0.01), 0.894844 (0.02) and 0.895522 (0.05). 0.005 is closest to 0.90, so the tie rule (equal to 4 decimals, then lower Winkler score) was not needed. All four are within 0.5 points of each other, and the choice sits at the lower edge of the grid. We did not extend the grid: it was fixed before the test months were scored, and changing it after seeing test results would be tuning on the test set. With gamma = 0.005, ACI's level for the 90% interval stayed between 0.0886 and 0.1078 (`results/tables/aci_alpha_path.csv`), so V4 ends up very close to V3.

**Bootstrap design.**
One list of the 365 forecast days in the test months and one seed (0) are used for every statistic, so every call resamples the same days: slices (months, cloud terciles) are computed on the resampled days' rows, and differences between variants are paired. Each statistic is a ratio of per-day sums (for example, covered hours over daylight hours), recomputed in each of the 1,000 replicates; the 95% interval runs from the 2.5th to the 97.5th percentile.

**Cloud-cover terciles cut at 0.076 and 0.696.**
These are the 1/3 and 2/3 quantiles of forecast total cloud cover over daylight rows (all zones) before 2013-07-01, so the cut points never depend on test data. In the test months they hold 4,372, 4,498 and 4,900 daylight hours (`results/tables/T3_by_cloud.csv`).

**Figure F2's legend sits below the panels**, so it does not hide any error bars.

## 2026-10-16

**Descriptive names.**
The baseline script and its outputs are named for what they contain (`scripts/run_baselines.py`, `results/baseline_scores.csv`, `results/baseline_summary.md`) rather than for when they were written.

**A from-scratch rerun reproduces every result.**
With `results/` emptied, `scripts/run_all.py` regenerated every CSV table and every figure byte for byte identical to the previous run, confirming that the fixed seeds and deterministic settings make the pipeline reproducible. The only difference was one column's alignment marker in `tuning_lightgbm.md`, from a change to the Markdown table helper; no number changed.
