import streamlit as st
import pandas as pd

from processors.demand import DemandCalculator
from processors.generation import GenerationCalculator
from processors.pricing import PricingCalculator
from processors.imbalance import ImbalanceCalculator
import plotly.express as px

from utils.file_loader import FileLoader
from utils.charts import ChartMaker
from utils.helpers import clean_df
from utils.helpers import clean_df, expand_monthly_to_hourly



# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Wheeling Model Dashboard",
    layout="wide",
)

# -----------------------------
# Centered Title + Subtitle
# -----------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="font-size: 4em; margin-bottom: 0;">ISMO Wheeling Dashboard</h1>
        <p style="font-size: 1.2em; color: #555;">Upload data and explore your wheeling economics interactively.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load utilities
loader = FileLoader()
charts = ChartMaker()

# -----------------------------
# Custom CSS for Centered Tabs
# -----------------------------
st.markdown(
    """
    <style>
    /* Center the tab container */
    div[data-baseweb="tab-list"] {
        justify-content: center !important;
    }

    /* Increase font size and spacing of tab labels */
    div[data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
    }

    /* Highlight the active tab */
    div[data-baseweb="tab"][aria-selected="true"] {
        color: #E53935 !important;  /* Highlight red */
        border-bottom: 3px solid #E53935 !important;
    }

    /* Soften inactive tabs */
    div[data-baseweb="tab"][aria-selected="false"] {
        color: #999 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Demand Input",
    "2. Generation Input",
    "3. Marginal Price",
    "4. Imbalance & Settlement"
])


# -----------------------------
# TAB 1 — DEMAND
# -----------------------------
with tab1:
    st.header("Demand Input: Multi-Scenario Comparison")

    st.markdown("Upload one or more hourly demand profiles to compare scenarios and view their calculated metrics.")

    # Upload multiple files
    col_a, col_b, col_c = st.columns(3)
    base_file = col_a.file_uploader("Base Case", type=["csv", "xlsx"], key="base_case")
    scen1_file = col_b.file_uploader("Scenario 1", type=["csv", "xlsx"], key="scenario1")
    scen2_file = col_c.file_uploader("Scenario 2", type=["csv", "xlsx"], key="scenario2")

    peak_demand = st.number_input("Peak Demand (MW)", min_value=0.0, step=0.1)

    # Store uploaded scenarios
    uploaded_scenarios = {
        "Base Case": base_file,
        "Scenario 1": scen1_file,
        "Scenario 2": scen2_file
    }

    scenario_dfs = {}
    scenario_results = {}

    # Process uploaded files
    for name, file in uploaded_scenarios.items():
        if file is not None:
            try:
                df = loader.load_file(file)
                df = clean_df(df)
                scenario_dfs[name] = df

                # Compute results using DemandCalculator
                demand_calc = DemandCalculator(df, peak_demand)
                scenario_results[name] = demand_calc.compute()

            except Exception as e:
                st.error(f"{name}: {e}")

    # Display metrics if we have results
    if len(scenario_results) > 0:
        st.subheader("Scenario Results")

        for name, results in scenario_results.items():
            st.markdown(f"### {name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Load Factor", f"{results['load_factor']:.2f}%")
            col2.metric("Annual Energy (GWh)", f"{results['annual_energy']:.2f}")
            col3.metric("Capacity Required (MW)", f"{results['capacity_required']:.2f}")
            st.markdown("---")

    # Proceed to graph comparison
    if len(scenario_dfs) > 0:
        first_df = next(iter(scenario_dfs.values()))
        if first_df.shape[1] == 12:
            months = first_df.columns.tolist()
            selected_month = st.selectbox("Select Month to View", months)

            # Combine data for selected month across all uploaded scenarios
            combined_df = pd.DataFrame({"Hour": range(1, len(first_df) + 1)})

            for name, df in scenario_dfs.items():
                if selected_month in df.columns:
                    combined_df[name] = df[selected_month]
                else:
                    st.warning(f"{name} does not contain column {selected_month}")

            # Plot multi-scenario comparison
            st.subheader("Demand Profile Comparison")
            fig = charts.multi_line(
                combined_df,
                x="Hour",
                y_columns=list(scenario_dfs.keys()),
                title=f"Hourly Demand Comparison — {selected_month}",
                y_label="Demand (MW)"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Uploaded data doesn’t appear to have 12 monthly columns.")


# =========================================================
# TAB 2 — GENERATION
# =========================================================
with tab2:
    st.header("Generation Input: Multi-Generator Analysis")

    st.markdown("Upload one or more generator profiles and specify parameters for each generator.")

    # Allow multiple generator uploads
    gen_files = st.file_uploader(
        "Upload generator profile files (CSV/Excel)",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    generator_data = []
    generator_results = []

    if gen_files:
        for i, gen_file in enumerate(gen_files):
            with st.expander(f"⚙️ Generator {i+1} Configuration"):
                try:
                    df = loader.load_file(gen_file)
                    df = clean_df(df)

                    st.write(f"**Preview:** {gen_file.name}")
                    st.dataframe(df.head())

                    # Extract generation column (numeric)
                    gen_cols = df.select_dtypes(include="number").columns.tolist()
                    gen_col = gen_cols[0] if gen_cols else None

                    # Inputs per generator
                    col1, col2, col3 = st.columns(3)
                    installed_capacity = col1.number_input(f"Installed Capacity (MW) — {gen_file.name}", min_value=0.0, step=0.5, key=f"cap_{i}")
                    cost_of_generation = col2.number_input(f"Cost of Plant (Billion PKR) — {gen_file.name}", min_value=0.0, step=0.1, key=f"cost_{i}")
                    eaf = col3.slider(f"EAF (Availability Factor) — {gen_file.name}", 0.0, 1.0, 0.45, 0.01, key=f"eaf_{i}")

                    # Compute generation metrics
                    gen_calc = GenerationCalculator(df, installed_capacity, cost_of_generation)
                    results = gen_calc.compute()

                    annual_gen = results["generation_gwh"]
                    gen_cost = results["generation_cost"]
                    effective_cap = installed_capacity * eaf

                    # Save data
                    generator_data.append({
                        "Name": gen_file.name,
                        "Installed Capacity (MW)": installed_capacity,
                        "Effective Capacity (MW)": effective_cap,
                        "Generation (GWh)": annual_gen,
                        "Cost (Rs/kWh)": gen_cost
                    })

                    # Heatmap Visualization
                    if "Day" in df.columns and "Hour" in df.columns and gen_col:
                        heatmap_fig = charts.heatmap(
                            df,
                            x="Hour",
                            y="Day",
                            z=gen_col,
                            title=f"{gen_file.name} — Day-Hour Generation Heatmap",
                            z_label="Generation (MW)"
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing {gen_file.name}: {e}")

        # Convert collected generator info into a DataFrame for summary charts
        summary_df = pd.DataFrame(generator_data)

        if not summary_df.empty:
            st.subheader("Summary Metrics")
            st.dataframe(summary_df.style.format({
                "Installed Capacity (MW)": "{:.2f}",
                "Effective Capacity (MW)": "{:.2f}",
                "Generation (GWh)": "{:.2f}",
                "Cost (Rs/kWh)": "{:.4f}"
            }))

            # Capacity vs Generation Chart
            cap_gen_df = summary_df.melt(
                id_vars=["Name"],
                value_vars=["Installed Capacity (MW)", "Effective Capacity (MW)", "Generation (GWh)"],
                var_name="Metric",
                value_name="Value"
            )

            st.subheader("Installed vs Effective Capacity vs Annual Generation")
            fig1 = charts.bar(cap_gen_df, x="Name", y="Value", title="Capacity & Generation Comparison")
            st.plotly_chart(fig1, use_container_width=True)

            # Cost Breakdown Chart
            st.subheader("Generation Cost Breakdown (Rs/kWh)")
            fig2 = px.bar(
                summary_df,
                x="Cost (Rs/kWh)",
                y="Name",
                orientation="h",
                title="Cost Comparison per Generator",
                text="Cost (Rs/kWh)",
                template="plotly_dark"
            )
            fig2.update_layout(title_x=0.5, margin=dict(t=80, b=50, l=80, r=40))
            st.plotly_chart(fig2, use_container_width=True)


# =========================================================
# TAB 3 — MARGINAL PRICE
# =========================================================
with tab3:
    st.header("Marginal Price Analysis")

    hmp_file = st.file_uploader("Upload Hourly Marginal Price (CSV/Excel)", type=["csv", "xlsx"])
    sensitivity = st.slider("Hourly Sensitivity Adjustment", min_value=-1.0, max_value=1.0, step=0.01, value=0.0)

    if hmp_file is not None:
        try:
            df = loader.load_file(hmp_file)
            df = clean_df(df)

            st.subheader("Preview")
            st.dataframe(df.head())

            # Compute Final Marginal Price
            pricing = PricingCalculator(df, sensitivity)
            df_fmp = pricing.compute()

            # Merge with original HMP for comparison
            numeric_col = df.select_dtypes(include=["number"]).columns[0]
            df_compare = pd.DataFrame({
                "Hour_Index": range(len(df_fmp)),
                "HMP": df[numeric_col].reset_index(drop=True),
                "FMP": df_fmp["FMP"]
            })

            # Overlay Line Chart (HMP vs FMP)
            st.subheader("Hourly Marginal Price vs Final Marginal Price")
            fig_compare = charts.multi_line(
                df_compare,
                x="Hour_Index",
                y_columns=["HMP", "FMP"],
                title="Hourly Marginal Price vs Adjusted Final Marginal Price",
                y_label="Price (Rs/kWh)"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Hourly Average Profile (24-hour mean)
            st.subheader("Average 24-Hour Price Profile")
            if "Hour" in df.columns and "Day" in df.columns:
                hourly_avg = df.groupby("Hour")[numeric_col].mean().reset_index()
                fig_avg = charts.line(hourly_avg, x="Hour", y=numeric_col, title="Average Hourly Price Profile", y_label="Rs/kWh")
                st.plotly_chart(fig_avg, use_container_width=True)

            # Price Distribution Histogram
            st.subheader("Price Distribution (Frequency of Occurrence)")
            fig_hist = charts.histogram(df, x=numeric_col, title="Distribution of Hourly Marginal Prices", x_label="Price (Rs/kWh)")
            st.plotly_chart(fig_hist, use_container_width=True)

            # Download Adjusted FMP
            st.subheader("Download Adjusted Final Marginal Price")
            st.download_button(
                label="Download Final Marginal Price CSV",
                data=df_fmp.to_csv(index=False).encode(),
                file_name="final_marginal_price.csv"
            )

        except Exception as e:
            st.error(str(e))


# =========================================================
# TAB 4 — IMBALANCE & SETTLEMENT
# =========================================================
with tab4:
    st.header("Imbalance & Settlement Dashboard")

    st.markdown(
        """
        Upload your demand, generation, and marginal price profiles to compute hourly imbalances,
        settlement amounts, and overall financial position.
        """
    )

    # Upload all required files
    col1, col2, col3 = st.columns(3)
    demand_file_imb = col1.file_uploader("Demand Profile", type=["csv", "xlsx"], key="demand_imb")
    gen_file_imb = col2.file_uploader("Generation Profile", type=["csv", "xlsx"], key="gen_imb")
    fmp_file_imb = col3.file_uploader("Final Marginal Price", type=["csv", "xlsx"], key="fmp_imb")

    # Hour selection
    hour = st.number_input("Inspect Hour Index (0–8759)", min_value=0, max_value=8759, value=0)

    if demand_file_imb and gen_file_imb and fmp_file_imb:
        try:
            demand_df = loader.load_file(demand_file_imb)
            gen_df = loader.load_file(gen_file_imb)
            fmp_df = loader.load_file(fmp_file_imb)

            # --- Expand monthly demand or generation if needed ---
            try:
                demand_series = expand_monthly_to_hourly(demand_df)
            except Exception:
                demand_series = demand_df.select_dtypes(include=["number"]).iloc[:, 0]

            try:
                gen_series = expand_monthly_to_hourly(gen_df)
            except Exception:
                gen_series = gen_df.select_dtypes(include=["number"]).iloc[:, 0]

            # --- Final Marginal Price (should already be hourly) ---
            price_series = fmp_df.select_dtypes(include=["number"]).iloc[:, 0]
            imbalance_calc = ImbalanceCalculator(demand_series, gen_series, price_series)

            # Compute hourly imbalance for full dataset
            df_all = pd.DataFrame({
                "Hour": range(len(demand_series)),
                "Demand (MW)": demand_series,
                "Generation (MW)": gen_series,
                "Price (Rs/kWh)": price_series
            })

            df_all["Purchase (MW)"] = (df_all["Demand (MW)"] - df_all["Generation (MW)"]).clip(lower=0)
            df_all["Sale (MW)"] = (df_all["Generation (MW)"] - df_all["Demand (MW)"]).clip(lower=0)
            df_all["Purchase (Rs)"] = (df_all["Purchase (MW)"] * df_all["Price (Rs/kWh)"]) / 1000
            df_all["Sale (Rs)"] = (df_all["Sale (MW)"] * df_all["Price (Rs/kWh)"]) / 1000
            df_all["Net (Rs)"] = df_all["Sale (Rs)"] - df_all["Purchase (Rs)"]

            # ===== Summary Statistics =====
            st.subheader("Summary Statistics")
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Total Purchase (MWh)", f"{df_all['Purchase (MW)'].sum():,.2f}")
            colB.metric("Total Sale (MWh)", f"{df_all['Sale (MW)'].sum():,.2f}")
            colC.metric("Total Purchase Cost (Rs)", f"{df_all['Purchase (Rs)'].sum():,.2f}")
            colD.metric("Total Sale Revenue (Rs)", f"{df_all['Sale (Rs)'].sum():,.2f}")
            st.metric("Net Settlement (Rs)", f"{df_all['Net (Rs)'].sum():,.2f}")

            st.divider()

            # ===== Visualization: Purchase vs Sale Imbalance =====
            st.subheader("Hourly Purchase vs Sale Imbalance (MW)")
            fig_imbalance = charts.multi_line(
                df_all,
                x="Hour",
                y_columns=["Purchase (MW)", "Sale (MW)"],
                title="Purchase vs Sale Imbalance (MW)",
                y_label="MW"
            )
            st.plotly_chart(fig_imbalance, use_container_width=True)

            # ===== Visualization: Settlement Trend =====
            st.subheader("Hourly Settlement (Rs)")
            fig_settle = charts.line(
                df_all,
                x="Hour",
                y="Net (Rs)",
                title="Net Settlement Over Time",
                y_label="Rs"
            )
            st.plotly_chart(fig_settle, use_container_width=True)

            # ===== Single-Hour Inspection =====
            st.subheader(f"Detailed View for Hour {hour}")
            result = imbalance_calc.compute_hour(hour)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Purchase Imbalance (MW)", f"{result['purchase_mw']:.5f}")
            col2.metric("Sale Imbalance (MW)", f"{result['sale_mw']:.5f}")
            col3.metric("Purchase Payment (MRs)", f"{result['m_rs']:.5f}")
            col4.metric("Sale Payment (MRs)", f"{result['n_rs']:.5f}")
            col5.metric("Net Payment (MRs)", f"{result['net_rs']:.5f}")

            # Optional download of imbalance dataset
            st.subheader("Download Imbalance Report")
            st.download_button(
                label="Download CSV",
                data=df_all.to_csv(index=False).encode(),
                file_name="imbalance_settlement_report.csv"
            )

        except Exception as e:
            st.error(str(e))

