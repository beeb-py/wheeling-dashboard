import pandas as pd
from typing import Dict, Any

class ImbalanceProcessor:
    """
    Computes hourly imbalance and settlement charges.
    Input: DataFrame with ['timestamp', 'scheduled_MW', 'actual_MW', 'price']
    """

    def __init__(self):
        pass

    def compute_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["imbalance_MW"] = df["actual_MW"] - df["scheduled_MW"]
        return df[["timestamp", "imbalance_MW"]]

    def settlement(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["imbalance_MW"] = df["actual_MW"] - df["scheduled_MW"]
        df["settlement_amount"] = df["imbalance_MW"] * df["price"]
        return df[["timestamp", "settlement_amount"]]

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "imbalance": self.compute_imbalance(df),
            "settlement": self.settlement(df)
        }


# Adapter for Streamlit app
class ImbalanceCalculator:
    """Compute per-hour imbalances and settlement for a single-hour query.

    Constructor accepts series-like inputs (pandas Series) for demand, generation and price.
    """

    def __init__(self, demand_series: pd.Series, gen_series: pd.Series, price_series: pd.Series):
        self.demand = demand_series.reset_index(drop=True).astype(float)
        self.gen = gen_series.reset_index(drop=True).astype(float)
        self.price = price_series.reset_index(drop=True).astype(float)

    def compute_hour(self, hour: int) -> Dict[str, float]:
        if hour < 0 or hour >= len(self.demand) or hour >= len(self.gen) or hour >= len(self.price):
            raise IndexError("Hour index out of range for provided series.")

        x = float(self.gen.iloc[hour])
        y = float(self.demand.iloc[hour])
        a = float(self.price.iloc[hour])

        purchase = max(y - x, 0.0)
        sale = max(x - y, 0.0)

        m = (purchase * a) / 1000.0
        n = (sale * a) / 1000.0

        net = n - m

        return {
            "purchase_mw": purchase,
            "sale_mw": sale,
            "m_rs": m,
            "n_rs": n,
            "net_rs": net
        }
