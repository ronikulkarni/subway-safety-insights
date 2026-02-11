
# Subway Safety Insights (Demo-Ready)

A lightweight, demo-ready Streamlit app that visualizes **hotspots, trends, and anomalies** for NYC Subway safety using public data. Ships with synthetic sample data so you can present **immediately**, and includes clear instructions to swap in **real NYPD + MTA datasets** later.

## Features (MVP)
- Map of station hotspots (incidents and rate per ridership).
- Time window filter and borough/station search.
- Trend lines with simple **anomaly detection** (z-score) to highlight spikes.
- Downloadable CSV of filtered results for sharing.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
or try this
```bash
(base) ronikulkarni@Ronis-MacBook-Pro Subway_Safety_Insights_With_Sim 2 % streamlit cache clear                         
(base) ronikulkarni@Ronis-MacBook-Pro Subway_Safety_Insights_With_Sim 2 % python3 -m pip install --upgrade pydeck==0.9.1
(base) ronikulkarni@Ronis-MacBook-Pro Subway_Safety_Insights_With_Sim 2 % python3 -m streamlit run app.py
```
The app loads sample CSVs from `data/` so it works instantly.

## Swap in real data (optional, after the demo)
1. **NYPD Complaint Data – Current/Year-to-Date**: Export CSV from NYC Open Data and filter to **Transit Bureau (T)** and offenses in subway stations. Keep columns: date (CMPLNT_FR_DT), hour (CMPLNT_FR_TM), offense (OFNS_DESC), latitude/longitude or station name, and precinct/transit district.
2. **Ridership**: Use **MTA Subway Hourly Ridership** or classic **Turnstiles** aggregation by station/complex. Compute a comparable **entries per week** metric.
3. **Stations**: Use **MTA Subway Stations (with GTFS Stop IDs)** and keep station name, lat, lon, borough, and GTFS stop id.
4. Save your cleaned files as:
   - `data/stations.csv`
   - `data/crime.csv` (columns: `gtfs_stop_id,date,incidents,category`)
   - `data/ridership.csv` (columns: `gtfs_stop_id,weekly_entries`)

The app will auto-detect and use real files if present.

## Talk track
- "Safety rate, not just counts": show **incidents per 1M entries** to normalize by station volume.
- "Actionable focus": a few stations contribute disproportionally to incidents—target them for focused patrols or design fixes.
- "Leading indicators": anomaly flags show **when** risk briefly spikes (events, construction, service changes).

## Folder
- `app.py` – Streamlit app
- `data/` – CSVs (sample + your real files later)
- `requirements.txt` – Python deps
- `README.md` – this file
