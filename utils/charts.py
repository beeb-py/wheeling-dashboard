import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class ChartMaker:
    """
    Simple wrapper around Plotly charts for interactive visualization.
    """

    def __init__(self, theme: str = "plotly_dark"):
        # You can change theme to "plotly_white" or others
        self.theme = theme

    # -----------------------------
    # LINE CHART
    # -----------------------------
    def line(self, df: pd.DataFrame, x: str, y: str, title: str = "", y_label: str = None):
        fig = px.line(
            df,
            x=x,
            y=y,
            title=title,
            markers=True,
            template=self.theme
        )
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y_label if y_label else y,
            title_x=0.5,  # Center title
            font=dict(size=14),
            hovermode="x unified"
        )
        return fig

    # -----------------------------
    # BAR CHART
    # -----------------------------
    def bar(self, df: pd.DataFrame, x: str, y: str, title: str = ""):
        fig = px.bar(
            df,
            x=x,
            y=y,
            title=title,
            template=self.theme
        )
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            title_x=0.5,
            font=dict(size=14)
        )
        return fig

    # -----------------------------
    # AREA CHART
    # -----------------------------
    def area(self, df: pd.DataFrame, x: str, y: str, title: str = ""):
        fig = go.Figure()

        # Add filled area trace
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y],
                mode="lines",
                fill="tozeroy",
                line=dict(color="#1f77b4"),
                name=y
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title=y,
            title_x=0.5,
            template=self.theme,
            font=dict(size=14),
            hovermode="x unified"
        )
        return fig

    # -----------------------------
    # MULTI-LINE CHART (Comparison)
    # -----------------------------
    def multi_line(self, df: pd.DataFrame, x: str, y_columns: list, title: str = "", y_label: str = None):
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly  # built-in color set

        for i, col in enumerate(y_columns):
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df[x],
                        y=df[col],
                        mode="lines+markers",
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,            # Center the title horizontally
                xanchor="center",
                yanchor="top",
                font=dict(size=18, color="#FFFFFF")  # optional styling
            ),
            xaxis_title=x,
            yaxis_title=y_label if y_label else "",
            template=self.theme,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.95,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80, b=40, l=60, r=60),
            font=dict(size=14)
        )
        return fig

    # -----------------------------
    # HEATMAP (Day-Hour Profile)
    # -----------------------------
    def heatmap(self, df: pd.DataFrame, x: str, y: str, z: str, title: str = "", z_label: str = "MW"):
        fig = go.Figure(
            data=go.Heatmap(
                x=df[x],
                y=df[y],
                z=df[z],
                colorscale="Viridis",
                colorbar=dict(title=z_label)
            )
        )
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(size=18)
            ),
            xaxis_title=x,
            yaxis_title=y,
            template=self.theme,
            margin=dict(t=100, b=60, l=60, r=60),
            font=dict(size=14)
        )
        return fig
    
    # -----------------------------
    # HISTOGRAM (Price Distribution with Range Labels)
    # -----------------------------
    def histogram(self, df: pd.DataFrame, x: str, title: str = "", x_label: str = None):
        import numpy as np

        # Automatically determine bin width
        values = df[x].dropna()
        nbins = 50  # adjustable granularity
        hist, bin_edges = np.histogram(values, bins=nbins)
        bin_labels = [f"{round(bin_edges[i],1)}–{round(bin_edges[i+1],1)}" for i in range(len(bin_edges)-1)]

        # Build new DataFrame for better labeling
        hist_df = pd.DataFrame({
            "Price Range (Rs/kWh)": bin_labels,
            "Frequency": hist,
            "Percentage": (hist / hist.sum()) * 100
        })

        # Create Plotly bar chart
        fig = px.bar(
            hist_df,
            x="Price Range (Rs/kWh)",
            y="Frequency",
            text=hist_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
            title=title,
            template=self.theme
        )

        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Hours: %{y}<br>Share: %{text}",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title=x_label if x_label else x,
            yaxis_title="Frequency (Hours)",
            title_x=0.5,
            font=dict(size=14),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        return fig

