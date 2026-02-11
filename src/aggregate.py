"""Aggregation helpers for weekly counts."""
import pandas as pd


def weekly_aggregate(events_df: pd.DataFrame) -> pd.DataFrame:
	grouped = (
		events_df.groupby(["week_id", "location_id", "syndrome_tag"]) 
		.size()
		.reset_index(name="count")
	)
	return grouped
