
#######################################################workingfine downnnn#########################################################
import sys
from pathlib import Path

# Add project root to sys.path
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import fetch_hourly_rides, fetch_predictions

# Setup
st.set_page_config(layout="wide")
st.title("!!!Citi Bike Prediction Monitor: Mean Absolute Error (MAE) Dashboard!!!")

# Sidebar
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,
    max_value=24 * 28,
    value=24,
    step=1,
)

# Fetch data
st.write("📥 Fetching Citi Bike predictions and actuals for the past", past_hours, "hours...")
df_actual = fetch_hourly_rides(past_hours)
df_pred = fetch_predictions(past_hours)

# Merge and calculate MAE
merged_df = pd.merge(df_actual, df_pred, on=["pickup_location_id", "pickup_hour"])
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

# ----- Overall MAE by Hour -----
mae_by_hour = (
    merged_df.groupby("pickup_hour")["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
)

st.subheader("📈 Overall MAE Over Time")
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error (MAE) for the Past {past_hours} Hours",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
    markers=True,
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

# ----- Top 3 Locations with Highest Avg MAE -----
mae_by_location = (
    merged_df.groupby("pickup_location_id")["absolute_error"]
    .mean()
    .reset_index()
    .rename(columns={"absolute_error": "MAE"})
    .sort_values("MAE", ascending=False)
)

top3_ids = mae_by_location.head(3)["pickup_location_id"].tolist()
top3_df = merged_df[merged_df["pickup_location_id"].isin(top3_ids)]

st.subheader("🔥 Top 3 Locations with Highest MAE")

fig_top3 = px.line(
    top3_df,
    x="pickup_hour",
    y="absolute_error",
    color="pickup_location_id",
    title="Top 3 Locations by MAE Over Time",
    labels={"pickup_hour": "Pickup Hour", "absolute_error": "MAE"},
    markers=True,
    template="plotly_dark",
)
st.plotly_chart(fig_top3, use_container_width=True)

# ----- Filter by Dropdown -----
st.sidebar.markdown("---")
all_location_ids = sorted(merged_df["pickup_location_id"].unique())
selected_ids = st.sidebar.multiselect("📍 Filter by Location ID", all_location_ids, default=top3_ids)

if selected_ids:
    selected_df = merged_df[merged_df["pickup_location_id"].isin(selected_ids)]
    st.subheader("🔍 Selected Locations: MAE Over Time")

    fig_filtered = px.line(
        selected_df,
        x="pickup_hour",
        y="absolute_error",
        color="pickup_location_id",
        title="Filtered Location(s) MAE Trend",
        labels={"pickup_hour": "Pickup Hour", "absolute_error": "MAE"},
        markers=True,
        template="plotly_dark",
    )
    st.plotly_chart(fig_filtered, use_container_width=True)
else:
    st.info("Please select at least one location from the dropdown to view its MAE trend.")

# ----- Summary -----
st.subheader("📊 MAE Summary Statistics")
col1, col2 = st.columns(2)
col1.metric("Average MAE", f"{merged_df['absolute_error'].mean():.3f}")
col2.metric("Max MAE", f"{merged_df['absolute_error'].max():.3f}")
col3, col4 = st.columns(2)
# col3.metric("Min MAE", f"{merged_df['absolute_error'].min():.3f}")
col4.metric("Standard Deviation of MAE", f"{merged_df['absolute_error'].std():.3f}")
st.write("📊 Summary statistics provide insights into the model's performance. A lower MAE indicates better predictive accuracy."
         )

# -------- Actual vs Predicted --------
# ----- Actual vs Predicted Rides Table -----
# st.subheader("📋 Actual vs Predicted Rides Table")

# display_table = available_names_df[
#     ["pickup_hour", "pickup_location_name", "rides", "predicted_demand", "absolute_error"]
# ].sort_values("pickup_hour", ascending=False).reset_index(drop=True)

# st.dataframe(display_table, use_container_width=True)
# Actual vs Predicted table
if selected_ids:
    # ... your existing MAE plot code ...

    st.subheader("📋 Actual vs Predicted Rides Table")
    st.dataframe(
        selected_df[["pickup_hour", "pickup_location_id", "rides", "predicted_demand", "absolute_error"]]
        .sort_values("pickup_hour", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )
else:
    st.info("Please select at least one location to view actual vs predicted rides.")
