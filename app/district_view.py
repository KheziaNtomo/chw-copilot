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
    "acute_watery_diarrhea": "Acute Watery Diarrhoea",
    "other":    "Other Syndromes",
    "unclear":  "Unclear Presentation",
}

SYNDROME_COLORS = {
    "respiratory_fever":     "#4a6032",
    "acute_watery_diarrhea": "#2e7d8a",
    "other":                 "#8a7a52",
    "unclear":               "#9a9a88",
}

# Consistent location colour palette
LOCATION_COLORS = [
    "#e07b54",  # burnt orange
    "#4a6032",  # olive
    "#2e7d8a",  # teal
    "#8a5a8a",  # muted purple
    "#b5770d",  # amber
    "#6b8e3d",  # lime
    "#c0392b",  # red
]

# Line dash patterns for distinguishing syndromes
SYNDROME_DASHES = {
    "respiratory_fever":     "solid",
    "acute_watery_diarrhea": "dash",
    "other":                 "dot",
    "unclear":               "dashdot",
}


def _detect_anomalies(weekly_df, baseline_weeks=4, threshold=2.0):
    """Compute anomalies dynamically from weekly data.

    For each (location, syndrome) pair, compute a rolling baseline mean
    over the preceding `baseline_weeks` weeks.  Flag any week where count
    >= threshold × baseline_mean *and* count >= 5.
    Returns a set of (week_id, location_id, syndrome_tag) tuples.
    """
    anomalies = set()
    for (loc, syn), grp in weekly_df.groupby(["location_id", "syndrome_tag"]):
        grp = grp.sort_values("week_id")
        counts = grp[["week_id", "count"]].values.tolist()
        for i, (wid, cnt) in enumerate(counts):
            # Need at least 2 prior weeks for a baseline
            if i < 2:
                continue
            start = max(0, i - baseline_weeks)
            baseline = [c for _, c in counts[start:i]]
            if not baseline:
                continue
            mean_val = sum(baseline) / len(baseline)
            if mean_val > 0 and cnt >= threshold * mean_val and cnt >= 5:
                anomalies.add((int(wid), loc, syn))
    return anomalies

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

    # Dynamic anomaly detection (computed on full data)
    detected_anomalies = _detect_anomalies(weekly_data)
    loc_id_to_name = {k: v.get("name", k) for k, v in DEMO_LOCATIONS.items()}
    sitrep = surv["sitrep"]

    # ── Week selector ─────────────────────────────────────────────
    all_week_ids = sorted(weekly_data["week_id"].unique())
    max_week = int(max(all_week_ids))
    wk_labels = {}
    for wid in all_week_ids:
        row = weekly_data[weekly_data["week_id"] == wid].iloc[0]
        wk_labels[wid] = row["week_label"]

    sel_col, _ = st.columns([1, 3])
    with sel_col:
        selected_week = st.selectbox(
            "Reporting week",
            options=list(reversed(all_week_ids)),
            format_func=lambda w: wk_labels.get(w, f"W{w}"),
            index=0,
            key="report_week",
        )

    latest_week = int(selected_week)
    latest = weekly_data[weekly_data["week_id"] == latest_week]
    total_cases = int(latest["count"].sum())

    # Build dynamic alerts for the selected week from anomaly detection
    alert_loc_ids = set()
    dynamic_alerts = []
    # Compute baselines for context
    for wid, loc_id, syn in detected_anomalies:
        if wid > latest_week:
            continue
        alert_loc_ids.add(loc_id)
        loc_name = loc_id_to_name.get(loc_id, loc_id)
        syn_display = SYNDROME_DISPLAY.get(syn, syn)
        # Get the actual count and baseline for this anomaly
        loc_syn_data = weekly_data[
            (weekly_data["location_id"] == loc_id) & (weekly_data["syndrome_tag"] == syn)
        ].sort_values("week_id")
        row_match = loc_syn_data[loc_syn_data["week_id"] == wid]
        count_val = int(row_match["count"].iloc[0]) if len(row_match) > 0 else 0
        # Compute baseline from prior 4 weeks
        prior = loc_syn_data[loc_syn_data["week_id"] < wid].tail(4)
        baseline_mean = prior["count"].mean() if len(prior) > 0 else 1
        ratio = count_val / baseline_mean if baseline_mean > 0 else 0
        # Severity based on ratio
        if ratio >= 5:
            severity = "critical"
        elif ratio >= 3:
            severity = "high"
        else:
            severity = "medium"
        # Latest-week count for this loc+syn (to show if resolved)
        latest_row = loc_syn_data[loc_syn_data["week_id"] == latest_week]
        latest_count = int(latest_row["count"].iloc[0]) if len(latest_row) > 0 else 0
        if wid == latest_week:
            trigger = "{} cases in W{} ({:.1f}x baseline)".format(count_val, wid, ratio)
        else:
            trigger = "Peaked at {} in W{} ({:.1f}x baseline), now {} in W{}".format(
                count_val, wid, ratio, latest_count, latest_week)
            if latest_count <= baseline_mean * 1.5:
                severity = "medium"  # Resolved
                trigger = "Resolved: " + trigger
        dynamic_alerts.append({
            "severity": severity,
            "location": loc_id,
            "location_name": loc_name,
            "syndrome": syn,
            "syndrome_display": syn_display,
            "message": trigger,
            "action": _recommend({"syndrome": syn}),
        })

    # Sort: critical first, then high, then medium
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    dynamic_alerts.sort(key=lambda a: sev_order.get(a["severity"], 9))

    num_alerts = len(dynamic_alerts)
    num_locations = latest["location_id"].nunique()
    weeks_tracked = latest_week - min(all_week_ids) + 1

    # ── Summary Metrics ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, str(total_cases),  f"Cases · Week {latest_week}", ""),
        (col2, str(num_alerts), "Active Alerts",  "alert" if num_alerts > 0 else "ok"),
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

    # -- Situation Report Table (dynamic alerts) --
    if dynamic_alerts:
        with st.expander(f"Situation Report ({num_alerts} alerts)", expanded=True):
            sev_colors = {"critical": "#a11", "high": "#c0392b", "medium": "#b5770d", "low": "#8D957E"}
            rows_html = ""
            for alert in dynamic_alerts:
                sev = alert["severity"]
                sev_c = sev_colors.get(sev, "#788990")
                loc = alert["location_name"]
                syn = alert["syndrome_display"]
                msg = alert.get("message", "")
                rec = alert.get("action", "")
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
        # Use dynamic anomaly detection for map
        alert_locations = alert_loc_ids

        # Build map data split into alert / normal
        alert_data = {"lats": [], "lons": [], "names": [], "hover": []}
        normal_data = {"lats": [], "lons": [], "names": [], "hover": []}
        for loc_id, loc_info in DEMO_LOCATIONS.items():
            loc_name = loc_info.get("name", loc_id)
            is_alert = loc_id in alert_locations

            loc_latest = latest[latest["location_id"] == loc_id]
            case_summary = " \u00b7 ".join(
                f"{row['syndrome_display']}: {int(row['count'])}"
                for _, row in loc_latest.iterrows()
            ) or "No data"

            status = "ALERT" if is_alert else "Normal"
            hover = (
                f"<b>{loc_name}</b><br>"
                f"Status: {status}<br>"
                f"Week {latest_week}: {case_summary}"
            )

            target = alert_data if is_alert else normal_data
            target["lats"].append(loc_info["lat"])
            target["lons"].append(loc_info["lon"])
            target["names"].append(loc_name)
            target["hover"].append(hover)

        map_fig = go.Figure()

        # Alert trace
        if alert_data["lats"]:
            map_fig.add_trace(go.Scattermapbox(
                lat=alert_data["lats"], lon=alert_data["lons"],
                mode="markers+text",
                marker=dict(size=22, color="#c0392b", opacity=0.85),
                text=alert_data["names"],
                textposition="top center",
                textfont=dict(size=11, color="#1e2a1e", family="Inter"),
                hovertext=alert_data["hover"],
                hoverinfo="text",
                name="Alert (outlier detected)",
            ))

        # Normal trace
        if normal_data["lats"]:
            map_fig.add_trace(go.Scattermapbox(
                lat=normal_data["lats"], lon=normal_data["lons"],
                mode="markers+text",
                marker=dict(size=14, color="#4a6032", opacity=0.85),
                text=normal_data["names"],
                textposition="top center",
                textfont=dict(size=11, color="#1e2a1e", family="Inter"),
                hovertext=normal_data["hover"],
                hoverinfo="text",
                name="Normal",
            ))

        all_lats = alert_data["lats"] + normal_data["lats"]
        all_lons = alert_data["lons"] + normal_data["lons"]
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)

        map_fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                y=1.02,
                x=0,
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(size=11, family="Inter"),
            ),
        )
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Syndrome Trends Line Chart (with date + illness toggles) ──────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'
        'font-weight:700;color:#8D957E;margin-bottom:0.5rem;">Weekly Syndrome Trends</p>',
        unsafe_allow_html=True,
    )

    if PLOTLY_AVAILABLE:
        # ── Assign consistent colours to locations ──
        all_locations = sorted(weekly_data["location_name"].unique())
        loc_color_map = {loc: LOCATION_COLORS[i % len(LOCATION_COLORS)] for i, loc in enumerate(all_locations)}
        # ── Filter controls: Date range + Syndrome + Location ──
        filter_row1, filter_row2, filter_row3 = st.columns([2, 2, 2])

        all_week_ids_chart = sorted(w for w in weekly_data["week_id"].unique() if w <= latest_week)
        min_week = int(min(all_week_ids_chart))
        max_week = int(max(all_week_ids_chart))

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

        with filter_row3:
            loc_options = ["Overall"] + all_locations
            selected_locations = st.multiselect(
                "Locations",
                options=loc_options,
                default=["Overall"],
                key="loc_select",
            )

            # Colour multiselect chips to match line colours
            if selected_locations:
                chip_css = ""
                for i, loc in enumerate(selected_locations):
                    c = loc_color_map.get(loc, "#444") if loc != "Overall" else "#444"
                    # Target the i-th tag inside THIS multiselect (last one on page)
                    chip_css += (
                        f'div[data-testid="stMultiSelect"]:last-of-type '
                        f'span[data-baseweb="tag"]:nth-of-type({i + 1}) '
                        f'{{ background-color: {c} !important; '
                        f'border-color: {c} !important; color: white !important; }}\n'
                    )
                st.markdown(f'<style>{chip_css}</style>', unsafe_allow_html=True)

        # Resolve "Overall" vs individual locations
        active_syns = selected_syndromes if selected_syndromes else all_syndromes
        use_overall = "Overall" in selected_locations
        individual_locs = [l for l in selected_locations if l != "Overall"]

        # Filter by date + syndrome
        chart_data = weekly_data[
            (weekly_data["week_id"] >= week_range[0]) &
            (weekly_data["week_id"] <= week_range[1]) &
            (weekly_data["week_id"] <= latest_week) &
            (weekly_data["syndrome_tag"].isin(active_syns))
        ]

        # Dynamic anomaly detection (filtered)
        anomaly_set = set()  # (week_id, location_name, syndrome_tag)
        for wid, loc_id, syn in detected_anomalies:
            ln = loc_id_to_name.get(loc_id, loc_id)
            if syn in active_syns:
                anomaly_set.add((wid, ln, syn))
        anomaly_week_ids = set()

        # Build week_id → week_label mapping
        wid_to_label = dict(zip(chart_data["week_id"], chart_data["week_label"]))

        fig = go.Figure()
        legend_shown = set()  # track which syndrome legends already shown

        # ── "Overall" line: sum across all locations ──
        if use_overall:
            overall_data = (
                chart_data
                .groupby(["week_id", "week_label", "syndrome_tag"])["count"]
                .sum()
                .reset_index()
                .sort_values("week_id")
            )
            for syndrome_tag in sorted(overall_data["syndrome_tag"].unique()):
                grp = overall_data[overall_data["syndrome_tag"] == syndrome_tag]
                syn_label = SYNDROME_DISPLAY.get(syndrome_tag, syndrome_tag)
                dash = SYNDROME_DASHES.get(syndrome_tag, "solid")
                show_legend = syndrome_tag not in legend_shown
                legend_shown.add(syndrome_tag)

                # Check if any location has anomaly this week for this syndrome
                m_colors, m_sizes = [], []
                for _, row in grp.iterrows():
                    wid = int(row["week_id"])
                    has_anom = any((wid, ln, syndrome_tag) in anomaly_set for ln in all_locations)
                    m_colors.append("#c0392b" if has_anom else "#444")
                    m_sizes.append(10 if has_anom else 6)
                    if has_anom:
                        anomaly_week_ids.add(wid)

                fig.add_trace(go.Scatter(
                    x=grp["week_label"], y=grp["count"],
                    mode="lines+markers",
                    name=syn_label,
                    legendgroup=syndrome_tag,
                    showlegend=show_legend,
                    line=dict(color="#444", width=3, dash=dash),
                    marker=dict(size=m_sizes, color=m_colors,
                                line=dict(color="white", width=1)),
                    hovertemplate=f"<b>Overall — {syn_label}</b><br>%{{x}} : %{{y}} cases<extra></extra>",
                ))

        # ── Per-location lines: colour = location, dash = syndrome ──
        for loc_name in sorted(individual_locs):
            loc_data = chart_data[chart_data["location_name"] == loc_name]
            color = loc_color_map.get(loc_name, "#888")
            for syndrome_tag in sorted(loc_data["syndrome_tag"].unique()):
                grp = loc_data[loc_data["syndrome_tag"] == syndrome_tag]
                grp = grp.groupby(["week_id", "week_label"])["count"].sum().reset_index().sort_values("week_id")

                syn_label = SYNDROME_DISPLAY.get(syndrome_tag, syndrome_tag)
                dash = SYNDROME_DASHES.get(syndrome_tag, "solid")
                show_legend = syndrome_tag not in legend_shown
                legend_shown.add(syndrome_tag)

                m_colors, m_sizes = [], []
                for _, row in grp.iterrows():
                    wid = int(row["week_id"])
                    is_anom = (wid, loc_name, syndrome_tag) in anomaly_set
                    m_colors.append("#c0392b" if is_anom else color)
                    m_sizes.append(11 if is_anom else 6)
                    if is_anom:
                        anomaly_week_ids.add(wid)

                fig.add_trace(go.Scatter(
                    x=grp["week_label"], y=grp["count"],
                    mode="lines+markers",
                    name=syn_label,
                    legendgroup=syndrome_tag,
                    showlegend=show_legend,
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(size=m_sizes, color=m_colors,
                                line=dict(color="white", width=1)),
                    hovertemplate=f"<b>{loc_name} — {syn_label}</b><br>%{{x}} : %{{y}} cases<extra></extra>",
                ))

        # Anomaly week shading
        all_chart_labels = sorted(
            chart_data["week_label"].unique(),
            key=lambda x: int(x.split("\u00b7")[0].strip()[1:]) if "\u00b7" in x else int(x.split("·")[0].strip()[1:])
        )
        for awid in sorted(anomaly_week_ids):
            wlabel = wid_to_label.get(awid)
            if wlabel and wlabel in all_chart_labels:
                idx = all_chart_labels.index(wlabel)
                fig.add_vrect(
                    x0=idx - 0.4, x1=idx + 0.4,
                    fillcolor="rgba(192,57,43,0.08)",
                    line=dict(color="rgba(192,57,43,0.3)", width=1, dash="dot"),
                    layer="below",
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
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#dde5d4",
                borderwidth=1,
                orientation="h",
                y=-0.2,
                font=dict(size=11),
            ),
            margin=dict(l=10, r=10, t=10, b=40),
            height=380,
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

