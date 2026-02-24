"""District View — Surveillance dashboard for district health officers."""
import io
import json
import streamlit as st
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

SYNDROME_DISPLAY = {
    "respiratory_fever":   "Respiratory Fever Syndrome",
    "acute_watery_diarrhea": "Acute Watery Diarrhea",
    "other":    "Other Syndromes",
    "unclear":  "Unclear Presentation",
}

SYNDROME_COLORS = {
    "respiratory_fever":     "#4a6032",
    "acute_watery_diarrhea": "#2e7d8a",
    "other":                 "#8a7a52",
    "unclear":               "#9a9a88",
}


def render_district_view():
    """Main district surveillance dashboard renderer."""
    from demo_data import DEMO_SURVEILLANCE, DEMO_LOCATIONS

    surv = DEMO_SURVEILLANCE

    # ── Page header ──────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="margin:0;font-size:2rem;font-weight:700;color:#1e2a1e;">
            District Surveillance
        </h1>
        <p style="color:#8a9a7a;margin:0.25rem 0 0 0;font-size:0.95rem;font-weight:400;">
            Syndromic surveillance · Anomaly detection · Situation reporting
        </p>
    </div>
    """, unsafe_allow_html=True)

    weekly_data = pd.DataFrame(surv["weekly_counts"])
    # Map location IDs to names
    weekly_data["location_name"] = weekly_data["location_id"].map(
        lambda lid: DEMO_LOCATIONS.get(lid, {}).get("name", lid)
    )
    weekly_data["syndrome_display"] = weekly_data["syndrome_tag"].map(
        lambda s: SYNDROME_DISPLAY.get(s, s)
    )

    latest_week  = int(weekly_data["week_id"].max())
    latest       = weekly_data[weekly_data["week_id"] == latest_week]
    total_cases  = int(latest["count"].sum())
    num_anomalies = len(surv["anomalies"])
    num_locations = latest["location_id"].nunique()
    weeks_tracked = weekly_data["week_id"].nunique()

    # ── Summary Metrics ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, str(total_cases),  f"Cases · Week {latest_week}", ""),
        (col2, str(num_anomalies), "Active Alerts",  "alert" if num_anomalies > 0 else "ok"),
        (col3, str(num_locations), "Locations Reporting", ""),
        (col4, str(weeks_tracked), "Weeks Tracked", ""),
    ]
    for col, val, label, cls in metrics:
        with col:
            value_cls = f' class="value {cls}"' if cls else ' class="value"'
            st.markdown(
                f'<div class="metric-card">'
                f'<div{value_cls}>{val}</div>'
                f'<div class="label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Anomaly Alerts ───────────────────────────────────────────
    if surv["anomalies"]:
        st.markdown(
            '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">Active Alerts</p>',
            unsafe_allow_html=True,
        )
        for anomaly in surv["anomalies"]:
            loc_name = DEMO_LOCATIONS.get(anomaly["location_id"], {}).get("name", anomaly["location_id"])
            syn_name = SYNDROME_DISPLAY.get(anomaly["syndrome_tag"], anomaly["syndrome_tag"])
            st.markdown(
                f'<div class="anomaly-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
                f'<div>'
                f'<div class="metric">{anomaly["current_count"]}</div>'
                f'<div style="font-weight:600;color:#1a2214;margin-top:0.25rem;">{syn_name}</div>'
                f'<div style="font-size:0.85rem;color:#8a9a7a;margin-top:0.15rem;">{loc_name}</div>'
                f'</div>'
                f'<div style="text-align:right;padding-top:0.25rem;">'
                f'<div style="font-size:1.8rem;font-weight:300;color:#c0392b;letter-spacing:-0.02em;">'
                f'{anomaly["ratio"]:.1f}×</div>'
                f'<div style="font-size:0.75rem;color:#8a9a7a;">above baseline ({anomaly["baseline_mean"]:.1f} avg)</div>'
                f'</div>'
                f'</div>'
                f'<div style="margin-top:0.6rem;font-size:0.88rem;color:#4a3030;border-top:1px solid rgba(192,57,43,0.15);padding-top:0.5rem;">'
                f'{anomaly["message"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Syndrome Trends Line Chart ───────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">Weekly Syndrome Trends</p>',
        unsafe_allow_html=True,
    )

    if PLOTLY_AVAILABLE:
        # Aggregate across all locations for overview line chart
        agg = (
            weekly_data
            .groupby(["week_id", "syndrome_tag", "syndrome_display"])["count"]
            .sum()
            .reset_index()
        )

        fig = go.Figure()
        for syndrome_tag in agg["syndrome_tag"].unique():
            grp = agg[agg["syndrome_tag"] == syndrome_tag].sort_values("week_id")
            color = SYNDROME_COLORS.get(syndrome_tag, "#888")
            label = SYNDROME_DISPLAY.get(syndrome_tag, syndrome_tag)
            fig.add_trace(go.Scatter(
                x=grp["week_id"],
                y=grp["count"],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color, line=dict(color="white", width=1.5)),
                hovertemplate=f"<b>{label}</b><br>Week %{{x}} : %{{y}} cases<extra></extra>",
            ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(247,247,244,0)",
            font=dict(family="Inter", color="#4a5a3a", size=12),
            xaxis=dict(
                title="Epidemiological Week",
                gridcolor="rgba(0,0,0,0.05)",
                linecolor="#dde5d4",
                tickmode="linear",
            ),
            yaxis=dict(
                title="Case Count",
                gridcolor="rgba(0,0,0,0.05)",
                linecolor="#dde5d4",
                rangemode="tozero",
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#dde5d4",
                borderwidth=1,
                orientation="h",
                y=-0.2,
            ),
            margin=dict(l=10, r=10, t=10, b=40),
            height=340,
        )
        fig.update_xaxes(showgrid=True, gridwidth=1)
        fig.update_yaxes(showgrid=True, gridwidth=1)

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.dataframe(weekly_data[["week_id","location_name","syndrome_display","count"]], use_container_width=True)

    # ── SITREP ───────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">Weekly Situation Report</p>',
        unsafe_allow_html=True,
    )

    sitrep = surv["sitrep"]
    narrative = sitrep["narrative"]

    # Parse narrative into styled sections
    sections = narrative.split("\n\n")
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Header line
        if section.startswith("WEEK") or section.startswith("week"):
            st.markdown(
                f'<div style="background:#1e2a1e;color:#fff;padding:0.8rem 1.2rem;'
                f'border-radius:8px;font-size:0.85rem;font-weight:600;letter-spacing:0.04em;'
                f'margin-bottom:0.75rem;">{section}</div>',
                unsafe_allow_html=True,
            )
        # Alert section
        elif section.startswith("ALERT"):
            st.markdown(
                f'<div style="background:rgba(192,57,43,0.06);border:1px solid rgba(192,57,43,0.15);'
                f'border-left:4px solid #c0392b;padding:0.8rem 1.2rem;border-radius:0 8px 8px 0;'
                f'font-size:0.9rem;line-height:1.7;color:#4a2020;margin-bottom:0.75rem;">'
                f'{section.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
        # Syndrome explanation
        elif "SYNDROME COVERS" in section or "WHAT THIS" in section:
            st.markdown(
                f'<div style="background:#f5f7f2;border:1px solid #dde5d4;'
                f'border-left:4px solid #4a6032;padding:0.8rem 1.2rem;border-radius:0 8px 8px 0;'
                f'font-size:0.88rem;line-height:1.8;color:#2d3d1f;margin-bottom:0.75rem;">'
                f'{section.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
        # Recommended actions
        elif "RECOMMENDED" in section or "ACTIONS" in section:
            st.markdown(
                f'<div style="background:rgba(74,96,50,0.05);border:1px solid #dde5d4;'
                f'border-left:4px solid #7a9e5a;padding:0.8rem 1.2rem;border-radius:0 8px 8px 0;'
                f'font-size:0.88rem;line-height:1.8;color:#2d3d1f;margin-bottom:0.75rem;">'
                f'{section.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
        # Data source / footer
        elif section.startswith("Data source"):
            st.markdown(
                f'<div style="font-size:0.75rem;color:#8a9a7a;padding:0.4rem 0;'
                f'border-top:1px solid #dde5d4;margin-top:0.5rem;">{section}</div>',
                unsafe_allow_html=True,
            )
        # Everything else
        else:
            st.markdown(
                f'<div style="background:#fff;border:1px solid #dde5d4;'
                f'padding:0.8rem 1.2rem;border-radius:8px;'
                f'font-size:0.88rem;line-height:1.8;color:#2d3d1f;margin-bottom:0.75rem;">'
                f'{section.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )

    if sitrep.get("alerts"):
        st.markdown(
            '<p style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
            'font-weight:700;color:#8a9a7a;margin:0.75rem 0 0.4rem 0;">Structured Alerts</p>',
            unsafe_allow_html=True,
        )
        for alert in sitrep["alerts"]:
            sev_color = {"high": "#c0392b", "medium": "#b5770d", "low": "#2e7d32"}.get(alert["severity"], "#888")
            loc_name = DEMO_LOCATIONS.get(alert["location"], {}).get("name", alert["location"])
            syn_name = SYNDROME_DISPLAY.get(alert["syndrome"], alert["syndrome"])
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:1rem;padding:0.6rem 1rem;'
                f'background:#fff;border:1px solid #dde5d4;border-left:4px solid {sev_color};'
                f'border-radius:0 8px 8px 0;margin:0.4rem 0;">'
                f'<span style="color:{sev_color};font-weight:700;font-size:0.7rem;'
                f'text-transform:uppercase;letter-spacing:0.08em;min-width:4rem;">{alert["severity"]}</span>'
                f'<span style="color:#1a2214;font-weight:600;">{loc_name}</span>'
                f'<span style="color:#8a9a7a;">·</span>'
                f'<span style="color:#4a5a3a;">{syn_name}</span>'
                f'<span style="color:#8a9a7a;margin-left:auto;font-size:0.85rem;">{alert["message"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


    # ── Export ───────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_csv, col_sitrep, _ = st.columns([1, 1, 2])
    with col_csv:
        csv_buf = io.StringIO()
        weekly_data[["week_id","location_name","syndrome_display","count"]].to_csv(csv_buf, index=False)
        st.download_button(
            "Download Counts (CSV)",
            csv_buf.getvalue(),
            file_name=f"chw_copilot_week{latest_week}_counts.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_sitrep:
        st.download_button(
            "Download SITREP (JSON)",
            json.dumps(sitrep, indent=2),
            file_name=f"chw_copilot_week{latest_week}_sitrep.json",
            mime="application/json",
            use_container_width=True,
        )
