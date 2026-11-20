# CowMonitor toolkit (Streamlit)

A small web app that runs the Zalliant 2024 capstone models on a CSV file you upload.
Upload one file, choose a module, press **Generate**, and the results, charts and
downloadable tables appear below. Nothing is saved: files stay in memory for your browser
session, and refreshing the page clears everything.

## Run it

Python 3.11 or newer; the test suite passes on 3.12 and 3.14.

```bash
cd streamlit_app
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1
```

The app opens at <http://127.0.0.1:8501>. Started this way, only this computer can reach it
(see [Privacy and safety](#privacy-and-safety) to share it on a network).

## Put it online (Streamlit Community Cloud)

1. Push this folder to a **private** GitHub repository. `requirements.txt` and
   `.streamlit/config.toml` must stay at the repository's top level.
2. On <https://share.streamlit.io>, sign in with that GitHub account and choose
   **Create app**: the repository, branch `main`, main file `app.py`, and Python 3.12 under
   **Advanced settings**. Then press **Deploy**.
3. An app from a private repository is private: only people you invite by email (the
   app's **Settings → Sharing** on share.streamlit.io) can open it, and they still need the passcode. Keep it that way; on the open internet
   the four-character passcode alone is easy to guess.

Every push to `main` updates the online app. Online, uploads are processed on Streamlit's
servers rather than this computer, and Community Cloud turns its usage statistics on. The app
hides the buttons Community Cloud adds to its toolbar: Share, star, edit and GitHub
(`_HIDE_CLOUD_TOOLBAR_BUTTONS` in `app.py`); the ⋮ menu stays. To invite viewers, use the
app's **Settings → Sharing** on share.streamlit.io instead of the hidden Share button.

## How to use it

0. **Type the 4-character passcode** into the four boxes, one letter or number per box (letters
   are case-sensitive); the app opens as soon as the last box is filled (pasting the code works
   too). It is set in `cowmonitor/access.py`. It is asked again
   after refreshing the page; **Start over** doesn't ask for it.
1. **Upload a CSV file.** The app recognises what it is from its contents and lists what it found:
   cows, readings and dates.
2. **Choose a module.** Only the modules that file can run are offered.
3. **Adjust settings if needed**, then press **Generate**.

The results show headline numbers, charts, tables with a **Download CSV** button, and,
except for the water intake estimate, a "How this is calculated" note. The farm time zone is in the sidebar (default US Eastern);
CowMonitor timestamps are UTC and are shown in that zone. **Start over** in the sidebar
clears the file and results.

## Files it accepts

| File | Where it comes from | What it needs |
|---|---|---|
| Temperature export | CowMonitor | A `timestamp` column (or separate `Date` and `Time` columns) and a `temperature` column, usually `deviceId` and optionally `Animal Id`. One cow or a whole herd; cows are told apart by device. Column names are matched loosely (`Device_ID`, `Temp`, ...). Timestamps in ISO format as CowMonitor exports them (`2024-04-06T04:01:55.000Z`), or numeric dates such as `04/06/2024 04:01`; dates that could be day/month or month/day are refused rather than guessed. |
| Drinking-removed copy (optional) | CowMonitor's processed data | The same cow and period with the drinking readings taken out. Used by the daily drinking summary and the water intake estimate. |
| Water temperature (optional) | A sensor in the trough the cow drinks from | `timestamp` and `temperature` columns, optionally `deviceId`, like a CowMonitor export; one sensor per file; 0–40 °C. Timestamps without a time zone are read as farm time. Only used by the water intake estimate, added in its settings. |
| Water-meter export | H2O Degree website | The hourly report list saved as CSV (no header; 23 columns per row). |

Limits per upload: 25 MB, 500,000 rows, 200 cows and 60 columns; bigger files are
refused with a message saying how to split them.

The project's own files work as examples: `../Data/Data Files/AbnormalData`,
`CalvingData`, and `DrinkingData` (`*_raw.csv`, `*_bouts.csv`, `2D.csv`, `2F.csv`). The
project has no water temperature readings, so `sample_data/synthetic_water_temperature_april2024.csv`
is a made-up file in the right format: 15 °C (the measured bowl temperature) every 15 minutes
for the project's April 2024 weeks. It shows how the upload works and gives the same results
as the 15 °C setting.

## Modules

For now the page offers only the **water intake estimate** and **daily water use**. The
other three are switched off: they are listed in `HIDDEN_MODULES` in `cowmonitor/modules.py`,
their code and tests stay, and taking a name out of that set offers it again.

| Module | Upload | What you get |
|---|---|---|
| Abnormal drinking check (model 1, switched off) | Temperature export | Normal / Abnormal for every cow, which rule fired, and the counts against their limits. |
| Calving alert (model 2, experimental, switched off) | Temperature export | Whether there is a calving-like temperature drop and when, whether the temperature rises again, and a half-day chart. |
| Daily drinking summary (switched off) | Temperature export, plus the drinking-removed copy if you have it | Drinking dips per day and the project's 8 daily drinking numbers. Exact with the copy, estimated without it. |
| Water intake estimate (experimental) | Temperature export, plus the drinking-removed copy and a water temperature file if you have them | Litres and gallons drunk per day and per drink by one cow, from its drinking dips and the water temperature (heat balance; see below). |
| Daily water use | Water-meter export | Gallons, flow events and minutes of flow per day, and the average by hour of day. Totals are for all cows on that meter together. |

Not included, because they need several files at once: the accuracy test (many labelled
files) and the project's water-intake regression (two cows' files plus the meter file). The
water intake estimate replaces the regression with a method that needs one cow's file.

## What changed from the notebooks, and how accurate it is

All results below are on the project's own test cows, the same ones the models were built
on, so accuracy on new cows is unknown.

- **Abnormal drinking check.** The rule is unchanged. With the original counts it gives the
  2024 notebook's exact results (21 of 21). Those counts were tuned on 6-day files, so
  shorter files look abnormal just for being short. The app always uses **per-day limits**,
  which scale them to the file's length: still 21 of 21 on the full files, and 20–21 of 21
  on 2- and 3-day slices, where the original counts got 14–19. Below one day there is no
  verdict. The original counts stay in the code only so the tests can check them against
  the notebook's results.
- **Calving alert.** The notebook compares each half-day with the wrong neighbours,
  treats missing half-days as 0 °C, and looks for the temperature rising again anywhere,
  not just after the drop. The app always uses a **fixed version** that corrects these. It
  was right for 12 of 16 test cows: it missed 3 of 7 calvings and raised 1 false alarm. The
  notebook's version (13 of 16, partly by luck) stays in the code only so the tests can
  check it against the notebook's results.
- **Daily drinking summary.** With CowMonitor's drinking-removed copy it reproduces the
  notebook's 8 features exactly. Without it, a reading more than 1.25 °C below the cow's
  median of the previous 12 hours counts as drinking. That estimate catches about 84% of
  CowMonitor's drinking readings, and about 83% of what it flags matches. Days use the
  farm time zone instead of a fixed UTC−4.
- **Water intake estimate (new).** Every litre a cow drinks has to be warmed from the
  water's temperature to the cow's, and the body supplies that heat while the rumen reading
  climbs back. So each drink's dip area (degrees below the cow's normal × hours, from the
  start of the drink until 2 hours later or the next drink) measures the heat, and
  `litres = K × area ÷ (normal − water temperature)`. The cow's normal is the 90th
  percentile of its readings over the surrounding 6 hours; times when it is under 35 °C (a
  bolus not yet in the cow) are left out. A day's litres are scaled up for missing
  readings, because a gap hides drinks; days under half covered, or with a drink that
  couldn't be estimated (water within 3 °C of the cow), get no estimate.
  - *Calibration.* K = 133.3 litres per hour makes the estimates add up to meters 2D and 2F
    (cows 1354 + 1568 and 1297 + 1709, 6–20 April 2024). Each meter report counts towards
    the hour it measured, and only complete meter days are compared (28; the export's last
    day lacks its final hour). On those days the daily totals came within 4.6 gal (about
    9%) on average, and 4.5 gal with dips estimated from the temperature alone. Calibrated
    on one meter and checked on the other: 4.6 gal a day. Leaving each day out in turn:
    4.8 gal, against 5.4 for guessing the average day. Hour by hour, estimate and meter
    correlate at r = 0.83.
  - *Water temperature.* The meters can't tell what temperature the water was: assuming
    anything from 0 to 30 °C and refitting K fits them equally well (daily error
    4.56–4.59 gal). K is therefore fitted at the temperature measured in the test pens'
    water bowls, 15 °C (59 °F), which is what the cows drank; the storage tank was 10 °C
    (50 °F), and the water warms on its way to the bowl. The estimate is only as right as
    that temperature: had the cows drunk water at the tank's 10 °C, K would be 161 and
    every estimate made with a real water temperature about 21% higher.
  - *Known limit.* Ordinary readings sit a little below the cow's normal, and each drink's
    window also collects that gap: about a quarter of a drink's area on these cows. K
    absorbs it on average, so the estimate partly follows how often a cow drinks: roughly
    15% high for a cow that sips far more often than these cows, and as much low for one
    that drinks far less often. Lower baselines and noise margins were tried and fitted the
    meters worse.
  - Recalibrate with `python -m tools.calibrate_intake` if the method or the reference water
    temperature (`REFERENCE_WATER_C` in `cowmonitor/intake.py`) changes; the tests check the
    stored figures against the meters. The tool assumes one water temperature for the whole
    period; using a logged water temperature for each drink would need a small extension.
- **Daily water use.** Days are full dates, not day-of-month, so a file spanning two
  months no longer merges the 6th of one month with the 6th of the next. Flow events and
  minutes of flow per day, partial-day flags and an hour-of-day profile are added.
- **Uploads.** Columns are found by name, not position, so the five column layouts in the
  delivered data all work. Rows with unreadable times or temperatures, readings outside
  10–45 °C, dates outside 2000–2099 and repeated readings are skipped, with a warning
  saying how many. A one-day file of 5-minute readings counts as exactly one day. A
  temperature file with fewer than 10% of its readings at 35 °C or more looks like water,
  and is refused in step 1 with a pointer to the water intake estimate; a water file whose
  median is 35 °C or more is refused as a cow's file. Water timestamps without a time zone
  are read as farm time, since trough loggers usually record local time.
- **Time zones.** Days follow the farm's calendar, including the 23- and 25-hour days when
  clocks change, so a complete day always shows 100% coverage.

## Privacy and safety

- Uploads are read in memory and never written to disk or cached between sessions.
- The app opens with a passcode screen; nothing else, sidebar included, shows until the
  passcode is entered. The passcode is fixed in `cowmonitor/access.py`, so it is a light lock,
  not real security: four characters are quick to guess, wrong tries aren't limited, and anyone who
  can read the code can see it.
- Started with `--server.address 127.0.0.1` as above, the app listens on this computer only,
  so other computers can't reach it. Without that option it listens on every network the
  computer is on. It has no user accounts of its own. To share it on a trusted network, run
  `streamlit run app.py --server.address 0.0.0.0`. For anything wider, deploy it as a private
  Community Cloud app ([Put it online](#put-it-online-streamlit-community-cloud)) or put it
  behind a proxy that adds a login and HTTPS.
- Uploads are treated as untrusted. Reading is bounded (the limits above), timestamps are
  parsed only in known formats, IDs from the file are reduced to letters, digits and
  `. _ : -`, and text from a file is shown literally (no links, images or formatting).
- CSV downloads quote every text cell and escape any that start like a spreadsheet formula.
- Streamlit's usage statistics are turned off when run locally, and error details stay in the
  terminal (online, in the app's logs).

## Development

```bash
pip install -r requirements-dev.txt
pytest                                   # all tests
pytest -m unit                           # synthetic data only
pytest -m integration                    # the project's CSVs (skipped if ../Data is missing)
pytest --cov --cov-report=term-missing     # cowmonitor/, app.py and tools/
black . && isort . && ruff check . && mypy cowmonitor app.py passcode_boxes.py tests tools && bandit -r cowmonitor app.py passcode_boxes.py tools
python -m tools.calibrate_intake         # recalibrate the water intake estimate on 2D and 2F
```

The tests check the app against the original code. `tests/fixtures` holds the 2024
notebook's saved outputs and the drinking features computed by the notebook's own
functions.

### Layout

```
app.py                 Streamlit page: passcode, upload, module choice, settings form, results
passcode_boxes.py      the four passcode boxes (a small Streamlit component: HTML, CSS, JS)
cowmonitor/            calculations (no Streamlit imports, enforced by ruff)
  access.py            the passcode
  loaders.py           read and recognise uploads; split cows
  abnormal.py          model 1
  calving.py           model 2 (fixed and as-delivered)
  drinking.py          daily drinking features (exact and estimated)
  intake.py            water intake estimate (heat balance)
  intake_calibration.py  compare the estimate with water meters; fit K
  water.py             daily water use from the meter export
  modules.py           the module list and how each builds its results
  charts.py            Plotly charts
  results.py           result containers and CSV export
  timezones.py         time-zone helpers
tools/calibrate_intake.py  calibration report for the water intake estimate
sample_data/           synthetic water temperature file (format example)
tests/                 pytest suite and reference fixtures
```
