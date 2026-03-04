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


def _recommend(alert):
    """Map syndrome to actionable recommendation for district leads."""
    syn = alert.get("syndrome", "")
    recs = {
        "respiratory_fever": "Verify cases; check ILI sample collection",
        "acute_watery_diarrhea": "Activate WASH response; collect stool samples",
        "other": "Review case definitions; investigate cluster",
        "unclear": "Clarify presentations; consider enhanced triage",
    }
    return recs.get(syn, "Investigate and verify counts")

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

    # Map week_id → date range labels using "date" column if available
    from datetime import date, timedelta
    EPI_START = date(2025, 1, 6)   # Monday of epi week 1

    def week_label(row):
        """Build 'W{n} · Mon DD–DD' label from date field or computed."""
        wid = int(row["week_id"])
        if "date" in row and row["date"]:
            try:
                start = date.fromisoformat(str(row["date"])[:10])
            except (ValueError, TypeError):
                start = EPI_START + timedelta(weeks=wid - 1)
        else:
            start = EPI_START + timedelta(weeks=wid - 1)
        end = start + timedelta(days=6)
        return f"W{wid} · {start.strftime('%b %d')}–{end.strftime('%d')}"

    weekly_data["week_label"] = weekly_data.apply(week_label, axis=1)

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

    # -- Situation Report Table (top of dashboard) --
    sitrep = surv["sitrep"]
    if sitrep.get("alerts"):
        with st.expander(f"Situation Report ({len(sitrep['alerts'])} alerts)", expanded=True):
            sev_colors = {"critical": "#a11", "high": "#c0392b", "medium": "#b5770d", "low": "#8D957E"}
            rows_html = ""
            for alert in sitrep["alerts"]:
                sev = alert["severity"]
                sev_c = sev_colors.get(sev, "#788990")
                loc = DEMO_LOCATIONS.get(alert["location"], {}).get("name", alert["location"])
                syn = SYNDROME_DISPLAY.get(alert["syndrome"], alert["syndrome"])
                msg = alert.get("message", "")
                rec = alert.get("action", _recommend(alert))
                rows_html += (
                    f'<tr>'
                    f'<td style="color:{sev_c};font-weight:700;text-transform:uppercase;'
                    f'font-size:0.7rem;letter-spacing:0.05em;white-space:nowrap;">{sev}</td>'
                    f'<td style="font-weight:600;color:#1a1510;">{loc}</td>'
                    f'<td style="color:#3a3225;">{syn}</td>'
                    f'<td style="color:#6a6255;font-size:0.85rem;">{msg}</td>'
                    f'<td style="color:#3a3225;font-size:0.85rem;">{rec}</td>'
                    f'</tr>'
                )
            th_style = ('text-align:left;padding:0.5rem 0.8rem;color:#6a6255;'
                       'font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;')
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
                f'<thead><tr style="border-bottom:1px solid rgba(227,214,197,0.2);">'
                f'<th style="{th_style}">Alert</th>'
                f'<th style="{th_style}">Location</th>'
                f'<th style="{th_style}">Syndrome</th>'
                f'<th style="{th_style}">Trigger</th>'
                f'<th style="{th_style}">Recommendation</th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table>'
                f'<style>'
                f'table tbody tr {{ border-bottom: 1px solid rgba(227,214,197,0.1); }}'
                f'table td {{ padding: 0.6rem 0.8rem; }}'
                f'</style>',
                unsafe_allow_html=True,
            )

    # ── Location Alert Map (ABOVE chart — shows alerts for current week) ───
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'font-weight:700;color:#8a9a7a;margin-bottom:0.5rem;">Location Alert Map</p>',
        unsafe_allow_html=True,
    )

    if PLOTLY_AVAILABLE and DEMO_LOCATIONS:
        # Determine which locations have anomalies
        alert_locations = set()
        for anomaly in surv["anomalies"]:
            alert_locations.add(anomaly["location_id"])

        # Build map data
        map_lats, map_lons, map_names, map_colors, map_sizes, map_hover = [], [], [], [], [], []
        for loc_id, loc_info in DEMO_LOCATIONS.items():
            loc_name = loc_info.get("name", loc_id)
            is_alert = loc_id in alert_locations

            # Compute latest-week case summary for hover
            loc_latest = latest[latest["location_id"] == loc_id]
            case_summary = " · ".join(
                f"{row['syndrome_display']}: {int(row['count'])}"
                for _, row in loc_latest.iterrows()
            ) or "No data"

            map_lats.append(loc_info["lat"])
            map_lons.append(loc_info["lon"])
            map_names.append(loc_name)
            map_colors.append("#c0392b" if is_alert else "#4a6032")
            map_sizes.append(22 if is_alert else 14)
            status = "ALERT" if is_alert else "Normal"
            map_hover.append(
                f"<b>{loc_name}</b><br>"
                f"Status: {status}<br>"
                f"Week {latest_week}: {case_summary}"
            )

        map_fig = go.Figure(go.Scattermapbox(
            lat=map_lats,
            lon=map_lons,
            mode="markers+text",
            marker=dict(size=map_sizes, color=map_colors, opacity=0.85),
            text=map_names,
            textposition="top center",
            textfont=dict(size=11, color="#1e2a1e", family="Inter"),
            hovertext=map_hover,
            hoverinfo="text",
        ))

        center_lat = sum(map_lats) / len(map_lats)
        center_lon = sum(map_lons) / len(map_lons)

        map_fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})

        # Legend
        st.markdown(
            '<div style="display:flex;gap:1.5rem;font-size:0.78rem;color:#4a5a3a;margin-top:0.3rem;">'
            '<span><span style="display:inline-block;width:10px;height:10px;background:#c0392b;'
            'border-radius:50%;margin-right:4px;"></span>Alert (outlier detected)</span>'
            '<span><span style="display:inline-block;width:10px;height:10px;background:#4a6032;'
            'border-radius:50%;margin-right:4px;"></span>Normal</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Syndrome Trends Line Chart (with date + illness toggles) ──────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'font-weight:700;color:#8D957E;margin-bottom:0.5rem;">Weekly Syndrome Trends</p>',
        unsafe_allow_html=True,
    )

    if PLOTLY_AVAILABLE:
        # ── Filter controls: Date range + Illness toggle ──
        filter_row1, filter_row2, filter_row3 = st.columns([2, 2, 2])

        all_week_ids = sorted(weekly_data["week_id"].unique())
        min_week = int(min(all_week_ids))
        max_week = int(max(all_week_ids))

        with filter_row1:
            week_range = st.slider(
                "Date range (epi weeks)",
                min_value=min_week,
                max_value=max_week,
                value=(min_week, max_week),
                key="week_range",
            )

        all_syndromes = sorted(weekly_data["syndrome_tag"].unique())
        with filter_row2:
            st.markdown(
                '<p style="font-size:0.7rem;color:#6a6255;margin-bottom:0.2rem;">Syndrome</p>',
                unsafe_allow_html=True,
            )
            selected_syndromes = []
            for syn in all_syndromes:
                label = SYNDROME_DISPLAY.get(syn, syn)
                if st.checkbox(label, value=True, key=f"syn_{syn}"):
                    selected_syndromes.append(syn)

        all_locations = sorted(weekly_data["location_name"].unique())
        with filter_row3:
            use_all = st.checkbox("All Locations", value=True, key="loc_all")
            if use_all:
                selected_locations = all_locations
            else:
                selected_locations = []
                for loc in all_locations:
                    if st.checkbox(loc, value=True, key=f"loc_{loc}"):
                        selected_locations.append(loc)
                st.markdown(
                    f'<span style="font-size:0.7rem;color:#8D957E;">{len(selected_locations)} of {len(all_locations)} selected</span>',
                    unsafe_allow_html=True,
                )

        # Apply filters
        chart_data = weekly_data[
            (weekly_data["week_id"] >= week_range[0]) &
            (weekly_data["week_id"] <= week_range[1]) &
            (weekly_data["syndrome_tag"].isin(selected_syndromes if selected_syndromes else all_syndromes)) &
            (weekly_data["location_name"].isin(selected_locations if selected_locations else all_locations))
        ]

        # Aggregate
        agg = (
            chart_data
            .groupby(["week_id", "week_label", "syndrome_tag", "syndrome_display"])["count"]
            .sum()
            .reset_index()
        )

        # Build week_id → week_label mapping for anomaly shading
        wid_to_label = dict(zip(agg["week_id"], agg["week_label"]))
        anomaly_week_ids = set(a["week_id"] for a in surv["anomalies"])

        fig = go.Figure()
        for syndrome_tag in agg["syndrome_tag"].unique():
            grp = agg[agg["syndrome_tag"] == syndrome_tag].sort_values("week_id")
            color = SYNDROME_COLORS.get(syndrome_tag, "#888")
            label = SYNDROME_DISPLAY.get(syndrome_tag, syndrome_tag)

            marker_colors = []
            marker_sizes = []
            marker_lines = []
            for _, row in grp.iterrows():
                if int(row["week_id"]) in anomaly_week_ids:
                    marker_colors.append("#c0392b")
                    marker_sizes.append(11)
                    marker_lines.append(dict(color="#c0392b", width=2))
                else:
                    marker_colors.append(color)
                    marker_sizes.append(7)
                    marker_lines.append(dict(color="white", width=1.5))

            fig.add_trace(go.Scatter(
                x=grp["week_label"],
                y=grp["count"],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=marker_sizes, color=marker_colors,
                            line=dict(color=[ml["color"] for ml in marker_lines],
                                      width=[ml["width"] for ml in marker_lines])),
                hovertemplate=f"<b>{label}</b><br>%{{x}} : %{{y}} cases<extra></extra>",
            ))

        # Red shaded regions for anomaly weeks
        all_chart_labels = sorted(agg["week_label"].unique(), key=lambda x: int(x.split("·")[0].strip()[1:]))
        for awid in sorted(anomaly_week_ids):
            wlabel = wid_to_label.get(awid)
            if wlabel and wlabel in all_chart_labels:
                idx = all_chart_labels.index(wlabel)
                fig.add_vrect(
                    x0=idx - 0.4, x1=idx + 0.4,
                    fillcolor="rgba(192,57,43,0.08)",
                    line=dict(color="rgba(192,57,43,0.3)", width=1, dash="dot"),
                    layer="below",
                    annotation_text="Alert" if awid == min(anomaly_week_ids) else "",
                    annotation_position="top left",
                    annotation_font=dict(color="#c0392b", size=10),
                )

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
                dtick=1,
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

