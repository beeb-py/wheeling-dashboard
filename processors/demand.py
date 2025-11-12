import pandas as pd
import numpy as np
from typing import Dict, Any

class DemandProcessor:
    """
    Processes system demand data.
    Input: DataFrame with columns ['timestamp', 'demand_MW', 'zone']
    """

    def __init__(self):
        pass

    def aggregate(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Total + zonal demand
        total = df.groupby("timestamp")["demand_MW"].sum().reset_index()
        total.rename(columns={"demand_MW": "total_demand_MW"}, inplace=True)

        zonal = df.groupby(["timestamp", "zone"])["demand_MW"].sum().reset_index()

        return {
            "total": total,
            "zonal": zonal
        }

    def peak_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        idx = df["demand_MW"].idxmax()
        return {
            "peak_timestamp": df.loc[idx, "timestamp"],
            "peak_demand_MW": df.loc[idx, "demand_MW"]
        }

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "aggregates": self.aggregate(df),
            "peak": self.peak_demand(df)
        }

# Backwards-compatible wrapper used by the Streamlit app
from utils.helpers import expand_30day_profile_to_year, mw_to_gwh
from utils.helpers import expand_24x12_profile_to_year

class DemandCalculator:
    """Simple adapter to provide compute(df, peak) interface expected by the app.

    compute() returns a dict with keys: load_factor, annual_energy, capacity_required
    """

    def __init__(self, df: pd.DataFrame, peak_demand: float):
        self.df = df.copy()
        self.peak_demand = float(peak_demand)

    def _get_series(self) -> pd.Series:
        # take first numeric column
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric column found in demand dataframe.")

        # If the numeric portion of the input looks like a 24x12 matrix (hours x months)
        # (or transposed), expand it into a full-year hourly profile using the helper.
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape == (24, 12) or numeric_df.shape == (12, 24):
            expanded_df = expand_24x12_profile_to_year(numeric_df)
            series = expanded_df["Base"].astype(float).reset_index(drop=True)
        else:
            series = self.df[num_cols[0]].astype(float).reset_index(drop=True)

        if len(series) == 24:
            # repeat daily profile for whole year
            series = pd.concat([series] * 365, ignore_index=True)
        elif len(series) == 720:
            # expand 30-day profile to year using helper
            expanded = expand_30day_profile_to_year(self.df[[num_cols[0]]], num_cols[0])
            series = expanded[num_cols[0]].astype(float).reset_index(drop=True)
        elif len(series) == 8760:
            series = series
        else:
            # accept other lengths by resampling/repeating/truncating to 8760 if possible
            if len(series) < 8760 and 8760 % len(series) == 0:
                reps = 8760 // len(series)
                series = pd.concat([series] * reps, ignore_index=True)
            else:
                raise ValueError(f"Unsupported demand series length: {len(series)}. Expected 24,720,8760 or a divisor of 8760.")

        return series

    def compute(self) -> Dict[str, Any]:
        series = self._get_series()

        annual_energy_gwh = mw_to_gwh(series)

        peak_possible_energy_mwh = self.peak_demand * 8760
        load_factor = (series.sum() / peak_possible_energy_mwh) * 100 if self.peak_demand > 0 else 0.0

        # capacity_required: for now return the provided peak_demand as placeholder
        capacity_required = float(self.peak_demand)

        return {
            "load_factor": load_factor,
            "annual_energy": annual_energy_gwh,
            "capacity_required": capacity_required
        }
