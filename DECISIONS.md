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

## 2026-10-17

**Follow-up fixes for conditional miscoverage: Mondrian CQR (V5) and recency-weighted CQR (V6).**
The results showed that CQR fixes average coverage but not coverage under clear versus cloudy forecasts, or month by month. We added two follow-up variants, labelled as follow-ups rather than part of the original design. Nothing in them is tuned on the test months. V5 computes one CQR adjustment per forecast-cloud tercile, with the tercile cut points fixed earlier from data before 2013-07-01; it has no free setting. V6 weights calibration day *d* days before the test month by 0.5^(d / half-life); the half-life was chosen from {7, 15, 30} days on development tasks 1-3 with the same rule as ACI's gamma, giving 7 days (90% coverage 0.899932 on tasks 1-3, `results/tables/weighted_half_life_selection.csv`). We report both whether they help or not. On the test months, V5 shrinks the 90% clear-minus-cloudy coverage gap from 0.0429 (CQR) to 0.0119 with a paired 95% interval of [-0.0075, 0.0311], and V6 does not reduce the mean monthly distance from 90% (0.0263 against 0.0220 for CQR) (`results/tables/T5_conditional.csv`). The development months show V5 narrowing the gap as well (0.0406 to 0.0240), while V6 had looked helpful there (0.0119 against 0.0175), which did not carry over (`results/tables/fixes_dev_check.csv`). For the development-month check only, the tercile cut points were computed from data that includes those months' forecast cloud cover (an input, never an outcome).

**Conditional coverage is summarised in T5.**
For each variant and level: the clear-minus-cloudy coverage gap with a paired day-bootstrap interval (both terciles computed on the same resampled days), the largest distance from the promise over the three terciles, and the mean and worst distance from the promise over the 12 months. These measure how evenly a variant keeps its promise, which yearly coverage alone cannot show.

**Clear-sky-normalised LightGBM added to Track 1.**
The farm locations are undisclosed, so a physical clear-sky model (which needs the sun's position) is impossible. We built an empirical envelope instead: for each zone, UTC hour and day of year, the 95th percentile of training power within 15 days of that date. The model predicts power divided by the envelope, with each hour weighted by its envelope so the loss equals the pinball loss on power, and adds the forecast clear-sky index and the transmittance (SSRD / TSR) as inputs. TSR in this dataset is net top-of-atmosphere radiation, which clouds reduce, so it is used as an input and not as a clear-sky reference. The envelope settings (0.95, 15 days, a floor of 0.01) were fixed in advance and not tuned, and the model reuses the frozen LightGBM settings. It scores 0.012107 against 0.012541 for plain LightGBM, better in 11 of the 12 months (`results/tables/T1_track1.csv`, `T1_track1_by_task.csv`). It produced 46 values above 1 (the envelope times a ratio above 1), which `finalize()` caps. Track 2 keeps plain LightGBM as its base model: that choice was fixed before any test result was seen, and switching to the best-scoring model afterwards would be choosing on the test set.

**Extended leakage check on every test month.**
Because the clear-sky model's average is level with the competition winner's reported 0.0121, which this project treats as a warning sign, every quantile model is now also leakage-checked on all 12 test months by `scripts/check_leakage.py`, part of `run_all.py`. It uses the five tuning levels to stay fast, because the check tests the code path, not accuracy. All checks pass (`results/tables/leakage_checks.csv`).

**Quantile crossing: report its size, not just its frequency.**
`finalize()` now records, before sorting, the largest drop between neighbouring levels in each daylight hour and how far sorting moves the values. For plain LightGBM the largest drop averaged 0.0358 of capacity (at most 0.157) and sorting moved values by 0.0072 on average; for linear QR, 0.0050 and 0.0005; for the clear-sky model, 0.0262 and 0.0045 (`results/tables/diagnostics_track1.csv`). Sorting (rearrangement) is kept as the fix; methods that avoid crossing by construction are described in the README but not implemented.

**Inference interface, packaging and continuous integration.**
`pvcal.forecast.forecast_month` and the `pvcal-forecast` command train on everything before a month and write its 99 quantiles; for June 2014 the command reproduces the Track 1 score of 0.0127. `pyproject.toml` (version 1.1.0) now carries the licence, author, readme, repository URL and the command's entry point, and builds a wheel. A GitHub Actions workflow runs pyflakes and the tests on Linux, Windows and macOS with Python 3.11 and 3.13; the real-data tests skip there because the data is not redistributed.

**Names and documentation.**
The repository is renamed `pv-interval-calibration` to match the study's subject and the package name `pvcal` ("PV calibration"). The README gives macOS and Linux setup commands first, with the Windows difference noted. `calibrate.py` explains each function in its docstring instead of commenting every line.
