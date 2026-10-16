# Getting the data

The raw competition data is not part of this repository (`data/raw/` is in `.gitignore`). Download it from the official source.

1. Get `GEFCom2014.zip`. The complete package is published as the appendix of Hong et al. (2016) on ScienceDirect. The competition organizer also shares it through a Dropbox link on the organizer's blog: https://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html (that post covers the load track, and the solar data is in the same package).
2. Unzip it. Inside the `GEFCom2014 Data` folder, unzip `GEFCom2014-S_V2.zip`.
3. Move the resulting `Solar` folder to `data/raw/Solar`, so that this file exists:

```
data/raw/Solar/Task 15/predictors15.csv
```

The expected layout:

```
data/raw/Solar/
  Instructions.txt
  Task 1/ … Task 15/            predictorsN.csv, trainN.csv, benchmarkNN.csv
  Solution to Task 15/Solution to Task 15.csv
```

Which files are used:
- `Task 15/predictors15.csv` holds the forecasts and observed power for the whole period, April 2012 to June 2014.
- The `benchmarkNN.csv` files hold the official benchmark.
- The solution file is only a fallback, for copies of the package whose `predictors15.csv` lacks June 2014 power.

To confirm the data is in the right place and matches what the code expects:

```bash
pytest tests/test_real_data.py -v
```

If you keep the data elsewhere, set `GEFCOM_SOLAR_DIR` for the tests and pass `--data` to the scripts.
