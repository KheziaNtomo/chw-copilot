"""District View — Surveillance dashboard for district health officers.

Features:
- Weekly syndrome counts dashboard (Plotly grouped bar chart)
- Anomaly alert cards with surge metrics
- SITREP viewer (narrative + structured data)
- CSV export for IDSR-compatible formats
- SITREP download as JSON
"""
import io
import json
import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_district_view():
    """Main district surveillance dashboard renderer."""
    from demo_data import DEMO_SURVEILLANCE, DEMO_LOCATIONS

    st.markdown("## 📊 District Surveillance Dashboard")
    st.markdown("Weekly syndromic surveillance with anomaly detection and SITREP generation.")

    surv = DEMO_SURVEILLANCE

    # ── Summary Metrics ──────────────────────────────────────
    weekly_data = pd.DataFrame(surv["weekly_counts"])
    latest_week = weekly_data["week_id"].max()
    latest = weekly_data[weekly_data["week_id"] == latest_week]
    total_cases = int(latest["count"].sum())
    num_anomalies = len(surv["anomalies"])
    num_locations = latest["location_id"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="value">{total_cases}</div>'
            f'<div class="label">Cases (Week {latest_week})</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="value" style="background:linear-gradient(135deg,#ef4444,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{num_anomalies}</div>'
            '<div class="label">Active Alerts</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="value">{num_locations}</div>'
            '<div class="label">Locations</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col4:
        weeks_tracked = weekly_data["week_id"].nunique()
        st.markdown(
            '<div class="metric-card">'
            f'<div class="value">{weeks_tracked}</div>'
            '<div class="label">Weeks Tracked</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Anomaly Alerts ───────────────────────────────────────
    if surv["anomalies"]:
        st.markdown("### 🚨 Active Anomaly Alerts")
        for anomaly in surv["anomalies"]:
            loc_name = DEMO_LOCATIONS.get(anomaly["location_id"], {}).get("name", anomaly["location_id"])
            st.markdown(
                f'<div class="anomaly-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div>'
                f'<div class="metric">{anomaly["current_count"]} cases</div>'
                f'<strong style="color:#f1f5f9">{anomaly["syndrome_tag"].replace("_"," ").title()}</strong>'
                f' at <strong style="color:#14b8a6">{loc_name}</strong>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="color:#ef4444;font-size:1.8rem;font-weight:700;">{anomaly["ratio"]:.1f}x</div>'
                f'<small style="color:#94a3b8">above baseline ({anomaly["baseline_mean"]:.1f} avg)</small>'
                f'</div>'
                f'</div>'
                f'<div style="margin-top:0.75rem;color:#f1f5f9;font-size:0.9rem">{anomaly["message"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Trends Chart ─────────────────────────────────────────
    st.markdown("### 📈 Weekly Syndrome Trends")

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            weekly_data,
            x="week_id",
            y="count",
            color="syndrome_tag",
            barmode="group",
            facet_col="location_id",
            labels={"week_id": "Epi Week", "count": "Cases", "syndrome_tag": "Syndrome"},
            color_discrete_map={
                "respiratory_fever": "#14b8a6",
                "acute_watery_diarrhea": "#3b82f6",
                "other": "#8b5cf6",
                "unclear": "#64748b",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=400,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(weekly_data, use_container_width=True)

    # ── SITREP ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Weekly SITREP")

    sitrep = surv["sitrep"]
    st.markdown(
        f'<div class="glass-card" style="white-space:pre-wrap;line-height:1.8;">'
        f'{sitrep["narrative"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if sitrep.get("alerts"):
        st.markdown("**Structured Alerts:**")
        for alert in sitrep["alerts"]:
            sev_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#3b82f6"}.get(alert["severity"], "#94a3b8")
            loc_name = DEMO_LOCATIONS.get(alert["location"], {}).get("name", alert["location"])
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:1rem;padding:0.5rem 1rem;'
                f'background:rgba(0,0,0,0.2);border-radius:8px;border-left:3px solid {sev_color};margin:0.5rem 0;">'
                f'<span style="color:{sev_color};font-weight:700;text-transform:uppercase;">{alert["severity"]}</span>'
                f'<span style="color:#f1f5f9">{loc_name} — {alert["syndrome"].replace("_"," ").title()}</span>'
                f'<span style="color:#94a3b8;margin-left:auto;">{alert["message"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Export Section ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export Data")

    col_csv, col_sitrep = st.columns(2)

    with col_csv:
        csv_buffer = io.StringIO()
        weekly_data.to_csv(csv_buffer, index=False)
        st.download_button(
            "📊 Download Weekly Counts (CSV)",
            csv_buffer.getvalue(),
            file_name=f"chw_copilot_week{latest_week}_counts.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_sitrep:
        sitrep_json = json.dumps(sitrep, indent=2)
        st.download_button(
            "📋 Download SITREP (JSON)",
            sitrep_json,
            file_name=f"chw_copilot_week{latest_week}_sitrep.json",
            mime="application/json",
            use_container_width=True,
        )
