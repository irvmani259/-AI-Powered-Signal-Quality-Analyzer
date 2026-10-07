"""
app/main.py
-----------
Streamlit dashboard for the AI-Powered Signal Quality Analyzer.

Run:
    streamlit run app/main.py

The dashboard talks directly to the Python modules (no HTTP hop needed)
so it works fully offline with the synthetic dataset.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ingestion import ingest, generate_synthetic
from detection import detect, DetectionResult
from forecasting import forecast, ForecastResult
from insights import generate_insights

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Signal Quality Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .severity-critical { color: #dc2626; font-weight: 700; }
    .severity-high     { color: #ea580c; font-weight: 700; }
    .severity-medium   { color: #d97706; font-weight: 600; }
    .severity-low      { color: #16a34a; font-weight: 600; }
    .severity-normal   { color: #6b7280; }
    .insight-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .recommendation-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key, default in [
    ("df", None),
    ("detection", None),
    ("forecast_result", None),
    ("insights", None),
    ("data_loaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Severity colour map
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "Critical": "#dc2626",
    "High":     "#ea580c",
    "Medium":   "#d97706",
    "Low":      "#16a34a",
    "Normal":   "#9ca3af",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80,
    )
    st.title("📡 Signal Quality\nAnalyzer")
    st.divider()

    st.header("📂 Data Source")
    data_mode = st.radio(
        "Choose input",
        ["Synthetic Dataset", "Upload CSV/JSON"],
        index=0,
    )

    if data_mode == "Synthetic Dataset":
        n_nodes = st.slider("Number of nodes", 2, 10, 5)
        hours = st.slider("Hours of data", 12, 168, 48)
        if st.button("🔄 Generate & Analyze", use_container_width=True, type="primary"):
            with st.spinner("Generating synthetic data…"):
                raw = generate_synthetic(n_nodes=n_nodes, hours=hours)
            with st.spinner("Ingesting & cleaning…"):
                st.session_state.df = ingest(raw)
            with st.spinner("Running detection…"):
                st.session_state.detection = detect(st.session_state.df)
            st.session_state.forecast_result = None
            st.session_state.insights = None
            st.session_state.data_loaded = True
            st.success(f"✅ {len(st.session_state.df):,} rows loaded")

    else:
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if uploaded and st.button("▶ Process File", use_container_width=True, type="primary"):
            fmt = "json" if uploaded.name.endswith(".json") else "csv"
            with st.spinner("Ingesting…"):
                st.session_state.df = ingest(uploaded.read(), fmt=fmt)
            with st.spinner("Running detection…"):
                st.session_state.detection = detect(st.session_state.df)
            st.session_state.forecast_result = None
            st.session_state.insights = None
            st.session_state.data_loaded = True
            st.success(f"✅ {len(st.session_state.df):,} rows loaded")

    st.divider()

    if st.session_state.data_loaded:
        st.header("⚙️ Forecast Settings")
        horizon_hours = st.slider("Forecast horizon (hours)", 1, 48, 12)
        if st.button("🔮 Run Forecast", use_container_width=True):
            with st.spinner("Forecasting… (may take ~30s)"):
                st.session_state.forecast_result = forecast(
                    st.session_state.df, horizon_hours=horizon_hours
                )
            st.success("Forecast complete")

        st.divider()
        st.header("🤖 AI Insights")
        if st.button("✨ Generate Insights", use_container_width=True):
            with st.spinner("Calling IBM Granite…"):
                st.session_state.insights = generate_insights(
                    anomaly_summary=st.session_state.detection.summary,
                    interference_events=st.session_state.detection.interference_events,
                    degradation_events=st.session_state.detection.degradation_events,
                    node_health=st.session_state.detection.node_health,
                )
            st.success("Insights generated")

    st.divider()
    st.caption("IBM Granite · watsonx.ai · FastAPI · Streamlit")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📡 AI-Powered Signal Quality Analyzer")

if not st.session_state.data_loaded:
    st.info(
        "👈 Use the sidebar to load a synthetic dataset or upload your own CSV/JSON file to begin."
    )
    st.markdown("""
**Expected CSV columns:**

| Column | Required | Description |
|---|---|---|
| `timestamp` | ✅ | ISO-8601 datetime |
| `node_id` | ✅ | Node identifier string |
| `snr` | ✅ | Signal-to-Noise Ratio (dB) |
| `ber` | ✅ | Bit Error Rate (0–1) |
| `latency` | ✅ | Round-trip latency (ms) |
| `frequency_band` | ☑️ | e.g. 1800MHz |
| `location` | ☑️ | e.g. SectorA |
| `packet_loss` | ☑️ | Packet loss (%) |
""")
    st.stop()

# ── Data references ────────────────────────────────────────────────────────
df: pd.DataFrame = st.session_state.df
det: DetectionResult = st.session_state.detection
fr: ForecastResult | None = st.session_state.forecast_result
insights: dict | None = st.session_state.insights

nodes = sorted(df["node_id"].unique().tolist())

# ===========================================================================
# TAB LAYOUT
# ===========================================================================
tab_overview, tab_timeseries, tab_anomalies, tab_heatmap, tab_forecast, tab_insights = st.tabs([
    "📊 Overview",
    "📈 Time Series",
    "⚠️ Anomalies",
    "🔥 Heatmaps",
    "🔮 Forecast",
    "🤖 AI Insights",
])

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================
with tab_overview:
    st.subheader("Dataset Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    total_rows = len(df)
    total_nodes = df["node_id"].nunique()
    anomaly_rows = len(det.anomalies)
    critical_rows = int((det.anomalies["severity"] == "Critical").sum()) if not det.anomalies.empty else 0
    anomaly_pct = round(100 * anomaly_rows / max(total_rows, 1), 1)

    c1.metric("Total Samples", f"{total_rows:,}")
    c2.metric("Monitored Nodes", total_nodes)
    c3.metric("Anomalous Rows", f"{anomaly_rows:,}", f"{anomaly_pct}%")
    c4.metric("Critical Events", critical_rows)
    c5.metric("Interference Events", len(det.interference_events))

    st.divider()

    col_left, col_right = st.columns(2)

    # Severity distribution pie
    with col_left:
        st.subheader("Severity Distribution")
        if not det.anomalies.empty:
            sev_counts = det.anomalies["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            fig_pie = px.pie(
                sev_counts,
                names="severity",
                values="count",
                color="severity",
                color_discrete_map=SEVERITY_COLORS,
                hole=0.4,
            )
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.success("✅ No anomalies detected")

    # Node health bar
    with col_right:
        st.subheader("Node Health Scores")
        if not det.node_health.empty:
            nh = det.node_health.sort_values("health_score")
            fig_bar = px.bar(
                nh,
                x="health_score",
                y="node_id",
                orientation="h",
                color="health_score",
                color_continuous_scale=["#dc2626", "#d97706", "#22c55e"],
                range_color=[0, 100],
                labels={"health_score": "Health Score (0–100)", "node_id": "Node"},
                text="health_score",
            )
            fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig_bar.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=40),
                height=320,
                xaxis_range=[0, 110],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Degradation & interference events
    if det.degradation_events or det.interference_events:
        st.divider()
        st.subheader("Detected Events")
        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            st.markdown("**📉 Degradation Events**")
            for ev in det.degradation_events:
                st.warning(ev["description"])
        with ev_col2:
            st.markdown("**📻 Interference Events**")
            for ev in det.interference_events:
                st.error(ev["description"])


# ===========================================================================
# TAB 2 — TIME SERIES
# ===========================================================================
with tab_timeseries:
    st.subheader("Signal Metrics Over Time")

    sel_nodes = st.multiselect("Select nodes", nodes, default=nodes[:3])
    metric_choice = st.selectbox("Metric", ["snr", "ber", "latency"], index=0)

    if sel_nodes:
        plot_df = df[df["node_id"].isin(sel_nodes)].copy()

        if metric_choice == "ber":
            plot_df["ber_log"] = np.log10(plot_df["ber"].clip(lower=1e-12))
            y_col = "ber_log"
            y_label = "BER (log₁₀)"
        else:
            y_col = metric_choice
            y_label = {"snr": "SNR (dB)", "latency": "Latency (ms)"}[metric_choice]

        fig_ts = px.line(
            plot_df,
            x="timestamp",
            y=y_col,
            color="node_id",
            labels={"timestamp": "Time", y_col: y_label, "node_id": "Node"},
        )

        # Overlay anomaly markers
        if not det.anomalies.empty:
            anom_df = det.anomalies[det.anomalies["node_id"].isin(sel_nodes)].copy()
            if not anom_df.empty:
                if metric_choice == "ber":
                    anom_df["plot_y"] = np.log10(anom_df["ber"].clip(lower=1e-12))
                else:
                    anom_df["plot_y"] = anom_df[metric_choice]

                fig_ts.add_trace(go.Scatter(
                    x=anom_df["timestamp"],
                    y=anom_df["plot_y"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=8,
                        color=anom_df["severity"].map(SEVERITY_COLORS),
                        line=dict(width=1),
                    ),
                    name="Anomaly",
                    hovertext=anom_df["severity"],
                ))

        # Threshold line
        thresh_map = {"snr": 10.0, "latency": 150.0}
        if metric_choice in thresh_map:
            fig_ts.add_hline(
                y=thresh_map[metric_choice],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({thresh_map[metric_choice]})",
                annotation_position="top right",
            )

        fig_ts.update_layout(height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig_ts, use_container_width=True)

        # Rolling mean overlay
        st.markdown("**Rolling Mean (window=15)**")
        roll_col = f"{metric_choice}_roll_mean_15"
        if roll_col in plot_df.columns:
            fig_roll = px.line(
                plot_df,
                x="timestamp",
                y=roll_col,
                color="node_id",
                labels={"timestamp": "Time", roll_col: f"{y_label} (roll avg)", "node_id": "Node"},
            )
            fig_roll.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("Select at least one node above.")


# ===========================================================================
# TAB 3 — ANOMALIES
# ===========================================================================
with tab_anomalies:
    st.subheader("Anomaly Details")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        fil_node = st.selectbox("Filter by node", ["All"] + nodes)
    with col_f2:
        fil_sev = st.selectbox("Filter by severity", ["All", "Critical", "High", "Medium", "Low"])
    with col_f3:
        max_rows = st.number_input("Max rows to show", 50, 2000, 200, step=50)

    anom_df = det.anomalies.copy()
    if fil_node != "All":
        anom_df = anom_df[anom_df["node_id"] == fil_node]
    if fil_sev != "All":
        anom_df = anom_df[anom_df["severity"] == fil_sev]

    anom_df = anom_df.head(int(max_rows))

    st.write(f"Showing **{len(anom_df)}** anomaly rows")

    if not anom_df.empty:
        display_cols = ["timestamp", "node_id", "snr", "ber", "latency", "severity", "severity_score"]
        display_cols = [c for c in display_cols if c in anom_df.columns]

        def _color_severity(val: str) -> str:
            colors = {
                "Critical": "background-color:#fee2e2",
                "High":     "background-color:#ffedd5",
                "Medium":   "background-color:#fef3c7",
                "Low":      "background-color:#dcfce7",
            }
            return colors.get(val, "")

        styled = anom_df[display_cols].style.applymap(
            _color_severity, subset=["severity"]
        ).format({"ber": "{:.2e}", "snr": "{:.2f}", "latency": "{:.1f}"})
        st.dataframe(styled, use_container_width=True, height=400)

        # Flag breakdown bar
        flag_cols = [c for c in anom_df.columns if c.startswith("flag_")]
        if flag_cols:
            flag_sums = anom_df[flag_cols].sum().sort_values(ascending=False)
            fig_flags = px.bar(
                x=flag_sums.index.str.replace("flag_", "", regex=False),
                y=flag_sums.values,
                labels={"x": "Flag type", "y": "Count"},
                title="Flag Type Breakdown",
                color_discrete_sequence=["#3b82f6"],
            )
            fig_flags.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_flags, use_container_width=True)
    else:
        st.success("No anomalies match the current filter.")


# ===========================================================================
# TAB 4 — HEATMAPS
# ===========================================================================
with tab_heatmap:
    st.subheader("🔥 Signal Quality Heatmaps")
    st.caption("All heatmaps use the full loaded dataset. Use the time-bucket slider to control resolution.")

    # ── shared controls ──────────────────────────────────────────────────────
    hm_col1, hm_col2 = st.columns([2, 1])
    with hm_col1:
        bucket_options = {"5 min": "5min", "15 min": "15min", "30 min": "30min", "1 hour": "1h", "3 hours": "3h", "6 hours": "6h"}
        bucket_label = st.select_slider(
            "Time bucket (resolution)",
            options=list(bucket_options.keys()),
            value="30 min",
        )
        bucket = bucket_options[bucket_label]
    with hm_col2:
        hm_nodes = st.multiselect("Filter nodes", nodes, default=nodes, key="hm_nodes")

    if not hm_nodes:
        st.warning("Select at least one node.")
        st.stop()

    hm_df = df[df["node_id"].isin(hm_nodes)].copy()
    hm_df["ts_bucket"] = hm_df["timestamp"].dt.floor(bucket)

    st.divider()

    # ── helper: pivot time × node ─────────────────────────────────────────────
    def _pivot(metric: str, agg: str = "mean") -> pd.DataFrame:
        """Return a DataFrame with time buckets as index, node_ids as columns."""
        return (
            hm_df.groupby(["ts_bucket", "node_id"])[metric]
            .agg(agg)
            .unstack("node_id")
            .sort_index()
        )

    # ─────────────────────────────────────────────────────────────────────────
    # HEATMAP 1 — SNR over time × node
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### 📶 SNR (dB) — Time × Node")
    st.caption("Green = good signal (high SNR), Red = degraded. Dashed threshold line at 10 dB.")

    snr_pivot = _pivot("snr")
    snr_labels = snr_pivot.round(1).astype(str)

    fig_snr_hm = go.Figure(go.Heatmap(
        z=snr_pivot.values,
        x=snr_pivot.columns.tolist(),
        y=[str(t)[:16] for t in snr_pivot.index],
        colorscale=[
            [0.0,  "#dc2626"],   # critical red   ≤ 5 dB
            [0.2,  "#ea580c"],   # high orange     ~10 dB
            [0.35, "#d97706"],   # medium amber    ~15 dB
            [0.55, "#84cc16"],   # acceptable      ~22 dB
            [1.0,  "#16a34a"],   # excellent green ≥ 35 dB
        ],
        zmin=0, zmax=40,
        text=snr_labels.values,
        texttemplate="%{text}",
        hovertemplate="Node: %{x}<br>Time: %{y}<br>SNR: %{z:.1f} dB<extra></extra>",
        colorbar=dict(title="SNR (dB)", thickness=14),
    ))
    fig_snr_hm.add_shape(
        type="line",
        x0=-0.5, x1=len(snr_pivot.columns) - 0.5,
        y0="threshold", y1="threshold",
        line=dict(color="white", width=1, dash="dot"),
    )
    fig_snr_hm.update_layout(
        title="SNR Heatmap — Time × Node",
        xaxis_title="Node",
        yaxis_title="Time bucket",
        height=max(320, min(60 * len(snr_pivot), 700)),
        margin=dict(t=50, b=30, l=120, r=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_snr_hm, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # HEATMAP 2 — BER over time × node  (log scale)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### 📉 BER (log₁₀) — Time × Node")
    st.caption("Blue = low BER (good), Red = high BER (degraded). Values shown as log₁₀(BER).")

    ber_pivot = _pivot("ber")
    ber_log_pivot = np.log10(ber_pivot.clip(lower=1e-12))
    ber_labels = ber_log_pivot.round(1).astype(str)

    fig_ber_hm = go.Figure(go.Heatmap(
        z=ber_log_pivot.values,
        x=ber_log_pivot.columns.tolist(),
        y=[str(t)[:16] for t in ber_log_pivot.index],
        colorscale=[
            [0.0,  "#1d4ed8"],   # excellent: very low BER
            [0.4,  "#60a5fa"],   # acceptable
            [0.65, "#fde68a"],   # marginal
            [0.82, "#f97316"],   # degraded
            [1.0,  "#dc2626"],   # critical: high BER
        ],
        zmin=-9, zmax=-2,
        text=ber_labels.values,
        texttemplate="%{text}",
        hovertemplate="Node: %{x}<br>Time: %{y}<br>log₁₀(BER): %{z:.2f}<extra></extra>",
        colorbar=dict(title="log₁₀(BER)", thickness=14),
    ))
    fig_ber_hm.update_layout(
        title="BER Heatmap (log scale) — Time × Node",
        xaxis_title="Node",
        yaxis_title="Time bucket",
        height=max(320, min(60 * len(ber_log_pivot), 700)),
        margin=dict(t=50, b=30, l=120, r=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_ber_hm, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # HEATMAP 3 — Latency over time × node
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### ⏱️ Latency (ms) — Time × Node")
    st.caption("Green = low latency, Red = high latency. Threshold at 150 ms.")

    lat_pivot = _pivot("latency")
    lat_labels = lat_pivot.round(0).astype(int).astype(str)

    fig_lat_hm = go.Figure(go.Heatmap(
        z=lat_pivot.values,
        x=lat_pivot.columns.tolist(),
        y=[str(t)[:16] for t in lat_pivot.index],
        colorscale=[
            [0.0,  "#16a34a"],
            [0.3,  "#84cc16"],
            [0.55, "#fde68a"],
            [0.75, "#f97316"],
            [1.0,  "#dc2626"],
        ],
        zmin=0, zmax=300,
        text=lat_labels.values,
        texttemplate="%{text}",
        hovertemplate="Node: %{x}<br>Time: %{y}<br>Latency: %{z:.0f} ms<extra></extra>",
        colorbar=dict(title="Latency (ms)", thickness=14),
    ))
    fig_lat_hm.update_layout(
        title="Latency Heatmap — Time × Node",
        xaxis_title="Node",
        yaxis_title="Time bucket",
        height=max(320, min(60 * len(lat_pivot), 700)),
        margin=dict(t=50, b=30, l=120, r=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_lat_hm, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # HEATMAP 4 — Anomaly density: hour-of-day × node
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### ⚠️ Anomaly Density — Hour of Day × Node")
    st.caption("Shows at which hours of the day each node experiences the most anomalies.")

    if not det.anomalies.empty:
        anom_hm = det.anomalies[det.anomalies["node_id"].isin(hm_nodes)].copy()
        anom_hm["hour"] = anom_hm["timestamp"].dt.hour
        density = (
            anom_hm.groupby(["hour", "node_id"])
            .size()
            .unstack("node_id")
            .reindex(index=range(24))
            .fillna(0)
        )

        fig_dens = go.Figure(go.Heatmap(
            z=density.values,
            x=density.columns.tolist(),
            y=[f"{h:02d}:00" for h in density.index],
            colorscale=[
                [0.0, "#f7f8fa"],
                [0.3, "#bfdbfe"],
                [0.6, "#f97316"],
                [1.0, "#dc2626"],
            ],
            hovertemplate="Node: %{x}<br>Hour: %{y}<br>Anomaly count: %{z}<extra></extra>",
            colorbar=dict(title="Count", thickness=14),
            text=density.values.astype(int),
            texttemplate="%{text}",
        ))
        fig_dens.update_layout(
            title="Anomaly Density by Hour of Day × Node",
            xaxis_title="Node",
            yaxis_title="Hour of Day",
            height=520,
            margin=dict(t=50, b=30, l=70, r=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_dens, use_container_width=True)
    else:
        st.success("✅ No anomalies detected — anomaly density heatmap not available.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # HEATMAP 5 — Severity distribution × node
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Severity Distribution — Node × Severity Level")
    st.caption("Count of rows at each severity level per node. Reveals which nodes are most troubled.")

    if not det.anomalies.empty:
        sev_order = ["Critical", "High", "Medium", "Low"]
        sev_hm = det.anomalies[det.anomalies["node_id"].isin(hm_nodes)].copy()
        sev_matrix = (
            sev_hm.groupby(["node_id", "severity"])
            .size()
            .unstack("severity")
            .reindex(columns=sev_order)
            .fillna(0)
        )

        fig_sev = go.Figure(go.Heatmap(
            z=sev_matrix.values,
            x=sev_matrix.columns.tolist(),
            y=sev_matrix.index.tolist(),
            colorscale=[
                [0.0,  "#f7f8fa"],
                [0.25, "#fef3c7"],
                [0.55, "#fed7aa"],
                [0.80, "#fca5a5"],
                [1.0,  "#dc2626"],
            ],
            hovertemplate="Node: %{y}<br>Severity: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count", thickness=14),
            text=sev_matrix.values.astype(int),
            texttemplate="%{text}",
        ))
        fig_sev.update_layout(
            title="Severity Distribution Heatmap — Node × Level",
            xaxis_title="Severity Level",
            yaxis_title="Node",
            height=max(280, 60 * len(sev_matrix) + 100),
            margin=dict(t=50, b=30, l=100, r=20),
        )
        st.plotly_chart(fig_sev, use_container_width=True)
    else:
        st.success("✅ No anomalies — severity heatmap not available.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # HEATMAP 6 — Node health metrics summary (original)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### 🏥 Node Health Metrics Summary")
    st.caption("Per-node aggregated health KPIs. Green = healthy, Red = degraded.")

    if not det.node_health.empty:
        nh = det.node_health[det.node_health["node_id"].isin(hm_nodes)].copy()
        metrics_for_heatmap = ["health_score", "anomaly_rate", "critical_pct", "mean_snr", "mean_latency"]
        metrics_for_heatmap = [m for m in metrics_for_heatmap if m in nh.columns]
        heat_data = nh.set_index("node_id")[metrics_for_heatmap].T

        # Normalise each row 0→1 for consistent colour across different scales
        heat_norm = heat_data.copy().astype(float)
        for row in heat_norm.index:
            rmin, rmax = heat_norm.loc[row].min(), heat_norm.loc[row].max()
            if rmax > rmin:
                heat_norm.loc[row] = (heat_norm.loc[row] - rmin) / (rmax - rmin)

        fig_hm = go.Figure(go.Heatmap(
            z=heat_norm.values,
            x=heat_norm.columns.tolist(),
            y=heat_norm.index.tolist(),
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            showscale=False,
            text=heat_data.round(1).values,
            texttemplate="%{text}",
            hovertemplate="Node: %{x}<br>Metric: %{y}<br>Value: %{text}<extra></extra>",
        ))
        fig_hm.update_layout(
            title="Node Health Metrics Heatmap (row-normalised)",
            xaxis_title="Node",
            yaxis_title="Metric",
            height=360,
            margin=dict(t=50, b=30, l=130, r=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Styled summary table
        st.dataframe(
            nh[["node_id", "health_score", "anomaly_rate", "mean_snr", "mean_ber", "mean_latency", "total_samples"]]
            .style
            .format({"mean_ber": "{:.2e}", "health_score": "{:.1f}", "anomaly_rate": "{:.1f}%"})
            .background_gradient(subset=["health_score"], cmap="RdYlGn")
            .background_gradient(subset=["anomaly_rate"], cmap="RdYlGn_r"),
            use_container_width=True,
        )


# ===========================================================================
# TAB 5 — FORECAST
# ===========================================================================
with tab_forecast:
    st.subheader("Signal Forecast & Failure Risk")

    if fr is None:
        st.info("👈 Click **Run Forecast** in the sidebar to generate predictions.")
    else:
        # Risk ranking table
        st.markdown("### ⚠️ Failure Risk Ranking")
        if not fr.risk_ranking.empty:
            rr = fr.risk_ranking.copy()
            rr["failure_probability"] = rr["failure_probability"].map("{:.1%}".format)
            rr["time_to_failure_hours"] = rr["time_to_failure_hours"].apply(
                lambda x: f"{x:.1f} h" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "—"
            )

            def _risk_style(val: str) -> str:
                return {
                    "Critical": "background-color:#fee2e2;font-weight:700",
                    "High":     "background-color:#ffedd5;font-weight:600",
                    "Medium":   "background-color:#fef3c7",
                    "Low":      "background-color:#dcfce7",
                }.get(val, "")

            st.dataframe(
                rr.style.applymap(_risk_style, subset=["risk_level"]),
                use_container_width=True,
            )

        # Forecast charts
        st.markdown("### 📈 Forecast Charts")
        fc_node = st.selectbox("Node to forecast", list(fr.forecasts.keys()))
        fc_metric = st.selectbox("Metric", ["snr", "ber", "latency"])

        node_fc = fr.forecasts.get(fc_node, {})
        metric_fc = node_fc.get(fc_metric)

        if metric_fc is not None and not metric_fc.empty:
            # Historical data
            hist = df[df["node_id"] == fc_node][["timestamp", fc_metric]].copy()
            hist = hist.rename(columns={"timestamp": "ds", fc_metric: "y"})
            if fc_metric == "ber":
                hist["y"] = np.log10(hist["y"].clip(lower=1e-12))
                metric_fc = metric_fc.copy()
                metric_fc["yhat"] = np.log10(metric_fc["yhat"].clip(lower=1e-12))
                metric_fc["yhat_lower"] = np.log10(metric_fc["yhat_lower"].clip(lower=1e-12))
                metric_fc["yhat_upper"] = np.log10(metric_fc["yhat_upper"].clip(lower=1e-12))
                y_label = "BER (log₁₀)"
            else:
                y_label = {"snr": "SNR (dB)", "latency": "Latency (ms)"}[fc_metric]

            fig_fc = go.Figure()
            # Historical
            fig_fc.add_trace(go.Scatter(
                x=hist["ds"], y=hist["y"],
                mode="lines", name="Historical",
                line=dict(color="#3b82f6", width=1.5),
            ))
            # Confidence band
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([metric_fc["ds"], metric_fc["ds"].iloc[::-1]]),
                y=pd.concat([metric_fc["yhat_upper"], metric_fc["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(251,146,60,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            ))
            # Forecast line
            fig_fc.add_trace(go.Scatter(
                x=metric_fc["ds"], y=metric_fc["yhat"],
                mode="lines", name="Forecast",
                line=dict(color="#ea580c", width=2, dash="dash"),
            ))
            # Threshold
            thresh_map = {"snr": 10.0, "latency": 150.0}
            if fc_metric in thresh_map:
                fig_fc.add_hline(
                    y=thresh_map[fc_metric],
                    line_dash="dot", line_color="red",
                    annotation_text="Failure threshold",
                )

            fig_fc.update_layout(
                title=f"{fc_node} — {fc_metric.upper()} Forecast",
                xaxis_title="Time",
                yaxis_title=y_label,
                height=420,
                legend=dict(orientation="h", y=-0.15),
                margin=dict(t=50, b=50),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Accuracy
            acc = fr.accuracy.get(fc_node, {}).get(fc_metric, {})
            if acc:
                ac1, ac2 = st.columns(2)
                ac1.metric("MAE (holdout)", f"{acc.get('mae', 'N/A')}")
                ac2.metric("RMSE (holdout)", f"{acc.get('rmse', 'N/A')}")
        else:
            st.warning(f"No forecast available for {fc_node} / {fc_metric}. Not enough historical data.")


# ===========================================================================
# TAB 6 — AI INSIGHTS
# ===========================================================================
with tab_insights:
    st.subheader("🤖 IBM Granite AI Insights")

    if insights is None:
        st.info("👈 Click **Generate Insights** in the sidebar to get AI-powered analysis.")
    else:
        source_badge = (
            "🟢 **IBM Granite via watsonx.ai**"
            if insights.get("source") == "granite"
            else "🟡 **Mock AI (no credentials configured)**"
        )
        st.markdown(source_badge)
        st.divider()

        # Root cause
        st.markdown("### 🔍 Root Cause Analysis")
        confidence = insights.get("confidence", 0)
        st.markdown(
            f'<div class="insight-box">'
            f'<strong>Root Cause:</strong> {insights.get("root_cause", "N/A")}<br>'
            f'<strong>Confidence:</strong> {confidence:.0%}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Confidence (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, 40],  "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100],"color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#1f2328", "width": 4},
                    "thickness": 0.75,
                    "value": confidence * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=20, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Plain summary
        st.markdown("### 📝 Engineer Summary")
        st.info(insights.get("plain_summary", ""))

        # Time window & affected nodes
        col_tw, col_an = st.columns(2)
        with col_tw:
            tw = insights.get("time_window", {})
            st.markdown("**⏱️ Time Window**")
            st.write(f"Start: `{tw.get('start', 'N/A')}`")
            st.write(f"End:   `{tw.get('end',   'N/A')}`")
        with col_an:
            st.markdown("**📡 Affected Nodes**")
            for n in insights.get("affected_nodes", []):
                st.write(f"• `{n}`")

        # Recommendations
        st.markdown("### ✅ Prioritized Recommendations")
        recs = insights.get("recommendations", [])
        if recs:
            for rec in sorted(recs, key=lambda r: r.get("priority", 99)):
                pri = rec.get("priority", "?")
                action = rec.get("action", "")
                rationale = rec.get("rationale", "")
                st.markdown(
                    f'<div class="recommendation-box">'
                    f'<strong>#{pri}</strong> {action}<br>'
                    f'<span style="color:#57606a;font-size:0.9em">📌 {rationale}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.write("No recommendations available.")
