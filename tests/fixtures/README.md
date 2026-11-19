# Test fixtures

Reference results produced by the original project code, used to check that the app
reproduces it.

| File | Source |
|---|---|
| `notebook_abnormal_2024.csv` | Saved output of `HealthEventsDetection.ipynb` cell 14 (`batchTest(root="AbnormalData", event="abnormal")`), as run by the project team in 2024 |
| `notebook_calving_2024.csv` | Saved output of `HealthEventsDetection.ipynb` cell 16 (`batchTest(root="CalvingData", event="calving")`) |
| `notebook_drinking_features_1354.csv`, `notebook_drinking_features_1297.csv` | `generateX(loadData(raw, bouts))` from `WaterIntakeReg.ipynb`, run unchanged on `DrinkingData/<cow>_raw.csv` and `<cow>_bouts.csv` |

The CSV data files themselves are not copied here; integration tests read them from
`../Data/Data Files` and are skipped when that folder is missing.
