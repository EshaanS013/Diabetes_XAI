# Data directory

Place the CDC BRFSS2015 **binary** CSV here:

`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

## Obtain the file

```bash
python scripts/download_dataset.py
```

If automatic download fails, download from Kaggle (`alexteboul/diabetes-health-indicators-dataset`) or the UCI Diabetes Health Indicators page and copy the binary CSV into `data/raw/`.

Then:

```bash
python -m src.data.prepare
```

Raw CSVs are gitignored due to size/licensing. Do not commit secrets.
