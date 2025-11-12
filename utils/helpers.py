import pandas as pd
import calendar
import numpy as np

def expand_30day_profile_to_year(df: pd.DataFrame, demand_column: str) -> pd.DataFrame:
    """
    Converts a 30-day hourly average profile (720 rows) into full-year 8760 rows
    by repeating the profile month-by-month.

    Assumes df has 24*30 rows in correct order.
    """

    if len(df) != 720:
        raise ValueError("Expected 720 rows (30 days * 24 hours).")

    # compute how many repeats are required to cover 8760 hours, then truncate
    import math
    reps = math.ceil(8760 / len(df))
    repeated = pd.concat([df] * reps, ignore_index=True)
    repeated = repeated.iloc[:8760].reset_index(drop=True)

    return repeated


def mw_to_gwh(series: pd.Series) -> float:
    """Convert MW hourly series into GWh/yr."""
    return series.sum() / 1000


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace, drops empty columns, fixes nulls."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")
    df = df.fillna(0)
    return df


def ensure_timestamp(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Forces timestamp dtype for a column."""
    df = df.copy()
    df[column] = pd.to_datetime(df[column])
    return df


def round_float(value: float, digits: int = 2) -> float:
    return round(value, digits)


def expand_24x12_profile_to_year(df: pd.DataFrame, year: int = 2023) -> pd.DataFrame:
    """
    Convert a 24x12 hourly-by-month profile into a full-year (8760 rows) hourly
    DataFrame with columns ['Fiscal Date', 'Day', 'Hour', 'Base'].

    The input `df` can be shaped either (24, 12) where each column is a month
    (January..December) and each row is hour 0..23, or transposed (12,24).

    Assumptions made:
    - Non-leap year (8760 hours).
    - Month order is the column order in `df` (assumed Jan..Dec if named accordingly).
    - Hours in rows are 0..23 (row 0 -> hour 0, will be output as Hour=1..24 for readability).
    """

    # normalize shape: we want hours in rows (24) and months in columns (12)
    _df = df.copy()
    if _df.shape == (12, 24):
        _df = _df.T

    if _df.shape != (24, 12):
        raise ValueError(f"Expected a 24x12 matrix (hours x months), got shape {_df.shape}")

    # number of days per month in a non-leap year
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    rows = []
    for month_idx, days in enumerate(month_days, start=1):
        # pick the column for this month (0-indexed)
        col = _df.iloc[:, month_idx - 1]

        # ensure we have 24 hourly values
        if len(col) != 24:
            raise ValueError(f"Month column {month_idx} does not contain 24 hourly values")

        for day in range(1, days + 1):
            for hour_idx in range(24):
                ts = pd.Timestamp(year=year, month=month_idx, day=day, hour=hour_idx)
                rows.append({
                    "Fiscal Date": ts,
                    "Day": int(ts.dayofyear),
                    "Hour": int(hour_idx + 1),
                    "Base": float(col.iloc[hour_idx])
                })

    result = pd.DataFrame(rows)
    # sanity: should be 8760 rows
    if len(result) != 8760:
        raise RuntimeError(f"Expanded profile length unexpected: {len(result)} rows (expected 8760)")

    return result

def expand_monthly_to_hourly(df: pd.DataFrame) -> pd.Series:
    """
    Expands a 12-column (monthly) x 24-row (hourly) DataFrame 
    into a full 8760-hour Series representing an average yearly profile.
    Each month's hourly average is repeated for the number of days in that month.
    """
    # Detect if DataFrame is in monthly format (12 columns, 24 rows)
    if df.shape[1] == 12 and df.shape[0] == 24:
        months = list(df.columns)
        days_in_month = [calendar.monthrange(2024, i + 1)[1] for i in range(12)]
        full_profile = []

        for i, month in enumerate(months):
            # Repeat each hour's average for all days in that month
            month_profile = np.tile(df[month].to_numpy(dtype=float), days_in_month[i])
            full_profile.extend(month_profile)

        # Return as a pandas Series with 8760 points
        return pd.Series(full_profile, name="Expanded_Profile")
    else:
        # If already hourly, return first numeric column
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric data found for expansion.")
        return df[numeric_cols[0]]
