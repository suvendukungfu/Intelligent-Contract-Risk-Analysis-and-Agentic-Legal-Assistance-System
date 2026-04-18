"""
analytics/dashboard.py
----------------------
Generates Plotly-based interactive charts for the Risk Analytics Dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

def create_confidence_trend_chart(ml_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates a Line Chart showing the model's confidence trend across the document clauses.
    """
    if not ml_results:
        return go.Figure()

    df = pd.DataFrame([{
        "Clause": r["clause_idx"] + 1,
        "Confidence": r["confidence"] * 100,
        "Risk": r["risk_level"]
    } for r in ml_results])

    # Assign colors based on risk
    color_map = {"High Risk": "#F87171", "Low Risk": "#4ADE80", "Unknown": "#8B949E"}
    
    fig = px.line(
        df, x="Clause", y="Confidence", 
        title="Model Confidence Trend Accross Document",
        markers=True,
        line_shape="spline",
        color_discrete_sequence=["#4F8EF7"] # Line color
    )
    
    # Overlay scatter to color nodes
    fig.add_trace(go.Scatter(
        x=df["Clause"], y=df["Confidence"],
        mode='markers',
        marker=dict(
            color=[color_map.get(r, "#8B949E") for r in df["Risk"]],
            size=10,
            line=dict(width=1, color="white")
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
        xaxis=dict(showgrid=False, title="Clause Number"),
        yaxis=dict(showgrid=True, gridcolor="#30363D", title="Confidence (%)", range=[0, 105]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_risk_distribution_chart(ml_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates a Donut Chart / Bar Chart for risk distribution.
    """
    if not ml_results:
        return go.Figure()

    df = pd.DataFrame([r["risk_level"] for r in ml_results], columns=["Risk Level"])
    counts = df.value_counts().reset_index(name='Count')
    
    color_map = {"High Risk": "#F87171", "Low Risk": "#4ADE80", "Unknown": "#8B949E"}

    fig = px.pie(
        counts, 
        names="Risk Level", 
        values="Count",
        hole=0.6,
        color="Risk Level",
        color_discrete_map=color_map,
        title="Risk Distribution"
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )
    return fig

def create_complexity_vs_risk_chart(ml_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates a Scatter plot comparing Clause Length (Complexity) vs Confidence, colored by Risk.
    Helps identify if long clauses are inherently riskier.
    """
    if not ml_results:
        return go.Figure()

    df = pd.DataFrame([{
        "Length (Words)": len(r["clause"].split()),
        "Confidence (%)": r["confidence"] * 100,
        "Risk": r["risk_level"],
        "Clause #": r["clause_idx"] + 1
    } for r in ml_results])

    color_map = {"High Risk": "#F87171", "Low Risk": "#4ADE80", "Unknown": "#8B949E"}

    fig = px.scatter(
        df, x="Length (Words)", y="Confidence (%)", 
        color="Risk", 
        color_discrete_map=color_map,
        size="Length (Words)", # Bubble size
        hover_data=["Clause #"],
        title="Clause Complexity vs Model Confidence"
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
        xaxis=dict(showgrid=True, gridcolor="#30363D"),
        yaxis=dict(showgrid=True, gridcolor="#30363D", range=[0, 105]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
