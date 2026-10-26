"""
Data and Model Visualization Module using Plotly.
Provides interactive figures for:
- Executive Dashboard (Distributions, Scatter, Bar)
- Model Comparison (Multi-metric comparison, Confusion matrix, ROC curves)
- SHAP Feature Contributions (Waterfall/Bar)
- What-If Risk Simulator (Gauges and Delta indicators)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List


# Color Palette
COLOR_LOW_RISK = "#2ECC71"   # Emerald Green
COLOR_HIGH_RISK = "#E74C3C"  # Crimson Red
COLOR_PRIMARY = "#3498DB"    # Professional Blue
COLOR_ACCENT = "#9B59B6"     # Purple
COLOR_BG = "#F8F9FA"


def plot_risk_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing Low Risk vs High Risk distribution."""
    risk_counts = df['risk'].value_counts().reset_index()
    risk_counts.columns = ['Risk', 'Count']
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts['Risk'],
        values=risk_counts['Count'],
        hole=0.55,
        marker=dict(colors=[COLOR_LOW_RISK if r == 'Low Risk' else COLOR_HIGH_RISK for r in risk_counts['Risk']]),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title_text="<b>Loan Applicant Risk Distribution</b>",
        title_font_size=16,
        showlegend=True,
        margin=dict(t=40, b=20, l=20, r=20),
        height=350
    )
    return fig


def plot_income_vs_loan(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of Income vs Loan Amount colored by Risk."""
    fig = px.scatter(
        df,
        x='income',
        y='loan_amount',
        color='risk',
        color_discrete_map={'Low Risk': COLOR_LOW_RISK, 'High Risk': COLOR_HIGH_RISK},
        hover_data=['credit_score', 'debt_to_income_ratio', 'employment_status'],
        labels={'income': 'Annual Income ($)', 'loan_amount': 'Loan Amount ($)', 'risk': 'Risk Category'},
        opacity=0.7
    )
    fig.update_layout(
        title_text="<b>Income vs. Loan Amount by Risk Status</b>",
        title_font_size=16,
        margin=dict(t=40, b=20, l=20, r=20),
        height=380
    )
    return fig


def plot_credit_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of Credit Scores broken down by Risk."""
    fig = px.histogram(
        df,
        x='credit_score',
        color='risk',
        barmode='overlay',
        nbins=35,
        color_discrete_map={'Low Risk': COLOR_LOW_RISK, 'High Risk': COLOR_HIGH_RISK},
        labels={'credit_score': 'Credit Score (FICO scale)', 'risk': 'Risk'}
    )
    fig.update_layout(
        title_text="<b>Credit Score Distribution by Risk</b>",
        title_font_size=16,
        margin=dict(t=40, b=20, l=20, r=20),
        height=350,
        xaxis=dict(range=[300, 850])
    )
    return fig


def plot_risk_by_employment(df: pd.DataFrame) -> go.Figure:
    """Stacked or grouped bar chart of risk proportion by employment status."""
    emp_risk = df.groupby(['employment_status', 'risk']).size().reset_index(name='count')
    
    fig = px.bar(
        emp_risk,
        x='employment_status',
        y='count',
        color='risk',
        barmode='group',
        color_discrete_map={'Low Risk': COLOR_LOW_RISK, 'High Risk': COLOR_HIGH_RISK},
        labels={'employment_status': 'Employment Status', 'count': 'Applicant Count', 'risk': 'Risk'}
    )
    fig.update_layout(
        title_text="<b>Applicant Risk by Employment Status</b>",
        title_font_size=16,
        margin=dict(t=40, b=20, l=20, r=20),
        height=350
    )
    return fig


def plot_model_comparison_metrics(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing all 6 models across 5 performance metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[metric],
            marker_color=colors[idx % len(colors)],
            text=[f"{v:.3f}" for v in comparison_df[metric]],
            textposition='auto'
        ))
        
    fig.update_layout(
        barmode='group',
        title_text="<b>Comparative Evaluation Across All Models</b>",
        title_font_size=16,
        yaxis=dict(title="Score", range=[0, 1.05]),
        xaxis=dict(title="Model Architecture"),
        margin=dict(t=40, b=40, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )
    return fig


def plot_interactive_confusion_matrix(cm: List[List[int]], model_name: str) -> go.Figure:
    """Plotly heatmap for confusion matrix."""
    labels = ['Low Risk', 'High Risk']
    z = cm
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=[[str(val) for val in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"<b>Confusion Matrix: {model_name}</b>",
        xaxis=dict(title="<b>Predicted Class</b>"),
        yaxis=dict(title="<b>Actual Class</b>", autorange="reversed"),
        margin=dict(t=40, b=20, l=20, r=20),
        width=400,
        height=380
    )
    return fig


def plot_interactive_roc_curves(metrics_dict: Dict[str, Any]) -> go.Figure:
    """Interactive multi-model ROC Curve overlay."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728', '#17becf']
    idx = 0
    
    for key, data in metrics_dict.items():
        name = data.get('model_name', key)
        roc = data.get('roc_curve', {})
        fpr = roc.get('fpr', [])
        tpr = roc.get('tpr', [])
        auc = data.get('roc_auc', 0.0)
        
        color = colors[idx % len(colors)]
        idx += 1
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f"{name} (AUC = {auc:.3f})",
            line=dict(color=color, width=2.5)
        ))
        
    # Baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Baseline (AUC = 0.500)',
        line=dict(color='gray', dash='dash', width=1.5)
    ))
    
    fig.update_layout(
        title="<b>Receiver Operating Characteristic (ROC) Comparison</b>",
        title_font_size=16,
        xaxis=dict(title="False Positive Rate", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", range=[0, 1.05]),
        margin=dict(t=40, b=40, l=20, r=20),
        legend=dict(x=0.45, y=0.08),
        height=480
    )
    return fig


def plot_shap_waterfall(contributions_df: pd.DataFrame, base_val: float, max_display: int = 10) -> go.Figure:
    """
    Renders an interactive waterfall/horizontal bar chart of individual SHAP drivers.
    Green = reduces High-Risk, Red = increases High-Risk.
    """
    df = contributions_df.head(max_display).copy().iloc[::-1]
    
    colors = [COLOR_HIGH_RISK if v > 0 else COLOR_LOW_RISK for v in df['shap_value']]
    
    fig = go.Figure(go.Bar(
        x=df['shap_value'],
        y=df['feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in df['shap_value']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="<b>Feature Impact on Loan Risk Prediction (SHAP Values)</b>",
        title_font_size=15,
        xaxis=dict(title="SHAP Value (Contribution to High-Risk Prediction)"),
        yaxis=dict(title="Feature"),
        margin=dict(t=40, b=20, l=150, r=40),
        height=420
    )
    return fig


def plot_what_if_comparison(
    prob_original: float,
    prob_new: float,
    orig_label: str,
    new_label: str
) -> go.Figure:
    """Side-by-side indicator gauges comparing original vs. modified applicant risk probability."""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=prob_new * 100,
        domain={'x': [0.55, 1], 'y': [0, 1]},
        delta={'reference': prob_original * 100, 'increasing': {'color': COLOR_HIGH_RISK}, 'decreasing': {'color': COLOR_LOW_RISK}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': COLOR_HIGH_RISK if prob_new >= 0.5 else COLOR_LOW_RISK},
            'steps': [
                {'range': [0, 50], 'color': "rgba(46, 204, 113, 0.2)"},
                {'range': [50, 100], 'color': "rgba(231, 76, 60, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': f"<b>Simulated Risk ({new_label})</b><br><span style='font-size:0.8em;color:gray'>New Scenario Probability</span>"}
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prob_original * 100,
        domain={'x': [0, 0.45], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': COLOR_HIGH_RISK if prob_original >= 0.5 else COLOR_LOW_RISK},
            'steps': [
                {'range': [0, 50], 'color': "rgba(46, 204, 113, 0.2)"},
                {'range': [50, 100], 'color': "rgba(231, 76, 60, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': f"<b>Original Risk ({orig_label})</b><br><span style='font-size:0.8em;color:gray'>Baseline Probability</span>"}
    ))
    
    fig.update_layout(
        margin=dict(t=50, b=20, l=20, r=20),
        height=280
    )
    return fig
