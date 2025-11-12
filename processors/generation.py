import pandas as pd
from typing import Dict, Any

class GenerationProcessor:
    """
    Processes generation fleet data.
    Input: DataFrame with ['timestamp', 'unit', 'gen_MW', 'fuel_type', 'available_MW']
    """

    def __init__(self):
        pass

    def total_generation(self, df: pd.DataFrame) -> pd.DataFrame:
        total = df.groupby("timestamp")["gen_MW"].sum().reset_index()
        total.rename(columns={"gen_MW": "total_gen_MW"}, inplace=True)
        return total

    def fuel_mix(self, df: pd.DataFrame) -> pd.DataFrame:
        mix = df.groupby("fuel_type")["gen_MW"].sum()
        mix = (mix / mix.sum() * 100).reset_index()
        mix.rename(columns={"gen_MW": "percent"}, inplace=True)
        return mix

    def availability_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["availability_factor"] = df["gen_MW"] / df["available_MW"]
        return df[["timestamp", "unit", "availability_factor"]]

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "totals": self.total_generation(df),
            "fuel_mix": self.fuel_mix(df),
            "availability": self.availability_factor(df)
        }


# Backwards-compatible adapter expected by the Streamlit app
import numpy as np

class GenerationCalculator:
    """Adapter providing compute() used by `app.py`.

    compute() returns dict with keys: generation_gwh, generation_cost
    """

    def __init__(self, df: pd.DataFrame, installed_capacity: float, cost_billion: float):
        self.df = df.copy()
        self.installed_capacity = float(installed_capacity)
        self.cost_billion = float(cost_billion)

    def _get_series(self) -> pd.Series:
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric column found in generation dataframe.")
        self._col_name = num_cols[0]
        return self.df[self._col_name].astype(float).reset_index(drop=True)

    def compute(self) -> Dict[str, Any]:
        series = self._get_series()

        # if short profile (24) repeat to year
        if len(series) == 24:
            series = pd.concat([series] * 365, ignore_index=True)
        elif len(series) == 720:
            from utils.helpers import expand_30day_profile_to_year
            expanded = expand_30day_profile_to_year(self.df[[self._col_name]], self._col_name)
            series = expanded[self._col_name].astype(float).reset_index(drop=True)
        elif len(series) == 8760:
            series = series
        else:
            if len(series) < 8760 and 8760 % len(series) == 0:
                reps = 8760 // len(series)
                series = pd.concat([series] * reps, ignore_index=True)
            else:
                raise ValueError(f"Unsupported generation series length: {len(series)}. Expected 24,720,8760 or a divisor of 8760.")

        generation_gwh = series.sum() / 1000.0

        if generation_gwh <= 0:
            generation_cost = float("inf")
        else:
            # cost_billion * 1000 / generation_gwh  (per user's contract)
            generation_cost = (self.cost_billion * 1000.0) / generation_gwh

        return {
            "generation_gwh": generation_gwh,
            "generation_cost": generation_cost
        }
