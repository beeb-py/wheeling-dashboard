import pandas as pd
import numpy as np
from typing import Dict, Any

class PricingProcessor:
    """
    Computes SMP and price sensitivities.
    Requires marginal cost curve or sorted generator stack.
    Input: DataFrame with ['unit', 'marginal_cost', 'gen_MW', 'max_MW']
    """

    def __init__(self):
        pass

    def compute_smp(self, stack: pd.DataFrame, demand: float) -> float:
        sorted_stack = stack.sort_values("marginal_cost")
        sorted_stack["cumulative_gen"] = sorted_stack["max_MW"].cumsum()

        marginal_unit = sorted_stack[sorted_stack["cumulative_gen"] >= demand].iloc[0]
        return marginal_unit["marginal_cost"]

    def price_sensitivity(self, stack: pd.DataFrame) -> float:
        # Simple approx: slope of marginal cost curve
        sorted_stack = stack.sort_values("marginal_cost")
        costs = sorted_stack["marginal_cost"].values
        mw = sorted_stack["max_MW"].values
        return np.gradient(costs, mw).mean()

    def run(self, stack_df: pd.DataFrame, demand_value: float) -> Dict[str, Any]:
        return {
            "SMP": self.compute_smp(stack_df, demand_value),
            "price_sensitivity": self.price_sensitivity(stack_df)
        }


# Adapter expected by the Streamlit app
class PricingCalculator:
    """Takes an HMP dataframe/series and a sensitivity scalar and returns FMP series/dataframe.

    compute() returns a DataFrame with column 'FMP'.
    """

    def __init__(self, df: pd.DataFrame, sensitivity: float = 0.0):
        self.df = df.copy()
        self.sensitivity = float(sensitivity)

    def _get_series(self) -> pd.Series:
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric column found in HMP dataframe.")
        self._col_name = num_cols[0]
        return self.df[self._col_name].astype(float).reset_index(drop=True)

    def compute(self) -> pd.DataFrame:
        series = self._get_series()

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
                raise ValueError(f"Unsupported HMP series length: {len(series)}. Expected 24,720,8760 or a divisor of 8760.")

        fmp = series * (1.0 + self.sensitivity)
        return pd.DataFrame({"FMP": fmp})
