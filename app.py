
import numpy as np
import streamlit as st
import os
import pandas as pd

import pydeck as pdk
from datetime import datetime

st.set_page_config(page_title="Subway Safety Insights", layout="wide")

@st.cache_data
def load_csv(path, required_cols=None):
    try:
        df = pd.read_csv(path)
        if required_cols:
            missing = set(required_cols) - set(df.columns)
            if missing:
                st.warning(f"{path} missing columns: {missing}. Using sample data instead.")
                return None
        return df
    except Exception as e:
        return None

# Auto-switch to real files if present
stations = load_csv("data/stations.csv") or load_csv("data/sample_stations.csv", ["gtfs_stop_id","station","lat","lon","borough"])
crime = load_csv("data/crime.csv") or load_csv("data/sample_crime.csv", ["gtfs_stop_id","date","incidents","category"])
ridership = load_csv("data/ridership.csv") or load_csv("data/sample_ridership.csv", ["gtfs_stop_id","weekly_entries"])

if stations is None or crime is None or ridership is None:
    st.error("Missing data files. Ensure CSVs exist in the data/ folder.")
    st.stop()

# Prep
crime["date"] = pd.to_datetime(crime["date"])
min_date, max_date = crime["date"].min(), crime["date"].max()

st.title("🗽 Subway Safety Insights (Demo)")
st.caption("Hotspots, trends, and anomalies — normalized by ridership")

with st.sidebar:
    st.header("Filters")
    d = st.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    if isinstance(d, list) and len(d)==2:
        start, end = pd.to_datetime(d[0]), pd.to_datetime(d[1])
    else:
        start, end = min_date, max_date

    boroughs = ["All"] + sorted(stations["borough"].dropna().unique().tolist())
    borough_choice = st.selectbox("Borough", boroughs, index=0)
    query = st.text_input("Station search", "")

# Join
base = crime.merge(stations, on="gtfs_stop_id", how="left").merge(ridership, on="gtfs_stop_id", how="left")
base = base[(base["date"]>=start) & (base["date"]<=end)].copy()

if borough_choice != "All":
    base = base[base["borough"]==borough_choice]

if query:
    base = base[base["station"].str.contains(query, case=False, na=False)]

# Aggregate by station
agg = base.groupby(["gtfs_stop_id","station","lat","lon","borough","weekly_entries"], as_index=False)["incidents"].sum()
agg["rate_per_million_entries"] = (agg["incidents"] / agg["weekly_entries"].replace(0, np.nan)) * 1_000_000

# Map
# Hotspots map — no JSON expressions
st.subheader("Hotspots map")

# Make a copy to avoid chained-assignment warnings
agg = agg.copy()

# Precompute everything we reference in deck.gl
safe_rate = agg["rate_per_million_entries"].fillna(0)

# Position as a single field
agg["position"] = agg[["lon", "lat"]].astype(float).values.tolist()

# Radius as a single numeric field
agg["radius"] = (safe_rate * 2 + 100).clip(lower=4, upper=10000).astype(float)

# Color as a single field [r,g,b,a]
r = np.minimum(255, (safe_rate / 2)).fillna(0).round().astype(int)
g = np.full(len(agg), 50, dtype=int)
b = np.full(len(agg), 80, dtype=int)
a = np.full(len(agg), 160, dtype=int)
# agg["fill_color"] = np.stack([r, g, b, a], axis=1).tolist()
# --- Bright, high-contrast color ramp (cyan -> yellow -> red) ---
# Normalize by 95th percentile so outliers don't wash out the scale
p95 = float(np.nanpercentile(safe_rate, 95)) if np.isfinite(safe_rate).any() else 1.0
p95 = p95 if p95 > 0 else 1.0
x = (safe_rate / p95).clip(0, 1).to_numpy()  # 0..1

# Piecewise linear ramp:
# 0.0  -> CYAN   (  0,180,255)
# 0.5  -> YELLOW (255,255,  0)
# 1.0  -> RED    (255,  0,  0)
c_cyan   = np.array([  0,180,255], dtype=float)
c_yellow = np.array([255,255,  0], dtype=float)
c_red    = np.array([255,  0,  0], dtype=float)

# Interpolate
first_half  = x <= 0.5
t1 = (x[first_half] * 2.0).reshape(-1, 1)                 # 0..1
t2 = ((x[~first_half] - 0.5) * 2.0).reshape(-1, 1)        # 0..1

colors = np.empty((len(x), 3), dtype=float)
# 0..0.5: cyan -> yellow
colors[first_half]  = c_cyan + (c_yellow - c_cyan) * t1
# 0.5..1: yellow -> red
colors[~first_half] = c_yellow + (c_red - c_yellow) * t2

# Build RGBA with strong alpha for contrast
alpha = 220
rgba = np.hstack([np.clip(colors, 0, 255).round().astype(int),
                  np.full((len(x), 1), alpha, dtype=int)])
agg["fill_color"] = rgba.tolist()

# Tooltip-safe rounded rate
agg["rate_rounded"] = safe_rate.round(1)

# View state
center_lat = float(agg["lat"].mean()) if len(agg) else 40.754
center_lon = float(agg["lon"].mean()) if len(agg) else -73.986
initial = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10.5)

# Layer: accessors reference single field names
layer = pdk.Layer(
    "ScatterplotLayer",
    data=agg,
    get_position="position",
    get_radius="radius",
    radius_min_pixels=4,
    radius_max_pixels=40,
    get_fill_color="fill_color",
    get_line_color=[255, 255, 255, 200],   # white stroke with some transparency
    line_width_min_pixels=1.5, 
    pickable=True,
)

st.pydeck_chart(
    pdk.Deck(
        initial_view_state=initial,
        layers=[layer],
        tooltip={"html": "<b>{station}</b><br/>Incidents: {incidents}<br/>Rate/Million: {rate_rounded}"}
    )
)

# Time series for selected station
st.subheader("Trends & anomalies")
col1, col2 = st.columns([2,1])
with col1:
    stations_list = ["All"] + agg["station"].dropna().sort_values().unique().tolist()
    station_choice = st.selectbox("Choose station", stations_list, index=0)
    ts = base.copy()
    if station_choice != "All":
        ts = ts[ts["station"]==station_choice]
    daily = ts.groupby("date", as_index=False)["incidents"].sum().sort_values("date")

    # simple anomaly detection (z-score)
    daily["z"] = (daily["incidents"] - daily["incidents"].rolling(14, min_periods=7).mean()) / (daily["incidents"].rolling(14, min_periods=7).std() + 1e-6)
    daily["anomaly"] = daily["z"].abs() > 2.0

    st.line_chart(daily.set_index("date")[["incidents"]])
    st.caption("Red dots mark statistically unusual days (|z|>2)")

    # Show anomalies
    anom = daily[daily["anomaly"]]
    if len(anom):
        st.warning(f"{len(anom)} anomaly day(s) detected in selection.")

with col2:
    st.metric("Stations in view", len(agg))
    st.metric("Incidents (period)", int(base["incidents"].sum()))
    total_entries = agg["weekly_entries"].sum()
    rate = (base["incidents"].sum()/total_entries*1_000_000) if total_entries else np.nan
    st.metric("Rate per 1M entries", f"{rate:.1f}" if pd.notna(rate) else "N/A")

st.subheader("Details")
st.dataframe(agg.sort_values("rate_per_million_entries", ascending=False).fillna(0))

# Download filtered data
csv = agg.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="subway_safety_filtered.csv", mime="text/csv")

st.info("Tip: Replace files in data/ with real exports named stations.csv, crime.csv, ridership.csv to go beyond the demo.")
