"""Deterministic anomaly detection per project rule.

For each (week_id, location_id, syndrome_tag) compute baseline_mean from the previous 4 calendar weeks (w-4 to w-1), filling missing weeks with 0.
Alert if count >= baseline_mean + 3 AND count >= 5. Suppress anything < 5.
"""
import pandas as pd


def detect_anomalies(weekly_counts: pd.DataFrame) -> pd.DataFrame:
	# expect columns: week_id (int), location_id (str), syndrome_tag (str), count (int)
	df = weekly_counts.copy()
	rows = []
	for (loc, tag), group in df.groupby(["location_id", "syndrome_tag"]):
		min_w = group["week_id"].min()
		max_w = group["week_id"].max()
		# Create full week sequence
		full_weeks = pd.DataFrame({"week_id": range(min_w, max_w + 1)})
		# Merge with actual counts, fill missing with 0
		merged = full_weeks.merge(group[["week_id", "count"]], on="week_id", how="left").fillna(0)
		# For each week, compute baseline from w-4 to w-1
		for _, row in merged.iterrows():
			w = row["week_id"]
			prev_weeks = [w - 4, w - 3, w - 2, w - 1]
			prev_counts = []
			for pw in prev_weeks:
				if pw >= min_w:
					p_row = merged[merged.week_id == pw]
					if not p_row.empty:
						prev_counts.append(p_row["count"].iloc[0])
			baseline_mean = float(pd.Series(prev_counts).mean()) if prev_counts else 0.0
			window_size = len(prev_counts)
			cur = row["count"]
			alert = cur >= 5 and cur >= baseline_mean + 3
			if alert:
				rows.append({
					"week_id": w,
					"location_id": loc,
					"syndrome_tag": tag,
					"count": cur,
					"baseline_mean": baseline_mean,
					"baseline_window_size": window_size
				})
	return pd.DataFrame(rows)
