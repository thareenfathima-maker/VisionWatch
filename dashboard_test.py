import streamlit as st
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
EVENT_FILE = PROJECT_ROOT / "events.csv"

st.set_page_config(
    page_title="VisionWatch",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ VisionWatch")
st.subheader("Factory Safety Monitoring Dashboard")

if EVENT_FILE.exists():
    df = pd.read_csv(EVENT_FILE)

    if df.empty:
        st.info("No safety events recorded yet.")

    else:
        # Summary metrics
        total_events = len(df)

        helmet_events = (
            df["event_type"] == "helmet_violation"
        ).sum()

        zone_events = (
            df["event_type"] == "restricted_zone_entry"
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Violations", total_events)
        col2.metric("Helmet Violations", helmet_events)
        col3.metric("Restricted-Zone Entries", zone_events)

        st.divider()

        st.subheader("Safety Event Records")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

else:
    st.warning("events.csv has not been created yet.")