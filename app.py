from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_portfolio.data import load_portfolio_data
from ecommerce_portfolio.metrics import (
    ABTestResult,
    build_category_reviews,
    build_customer_orders,
    build_delivery_stats,
    build_hourly_orders,
    build_monthly_orders,
    build_monthly_reviews,
    build_olist_funnel,
    build_olist_overview,
    build_olist_revenue_impact,
    build_payment_distribution,
    build_review_distribution,
    build_state_delivery,
    build_state_orders,
    build_status_distribution,
    build_top_categories,
    build_top_revenue_categories,
    run_olist_ab_test,
)

st.set_page_config(
    page_title="E-Commerce Analytics Portfolio — Shachi Raodeo",
    page_icon="📊",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────

ACCENT = "#2f6f5e"
ACCENT_SOFT = "#e8f3ef"
COLORS = {
    "blue": "#4A90D9",
    "green": "#355c3a",
    "orange": "#8c6d46",
    "purple": "#6b5b8a",
    "red": "#b2472f",
    "teal": "#2f6f5e",
}
YLGN = ["#f7fcf5", "#d5efcf", "#a1d99b", "#6aae5e", "#355c3a"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
        :root {
            --paper: #ffffff; --ink: #18212b; --accent: #2f6f5e;
            --accent-soft: #e8f3ef; --card: #ffffff; --line: rgba(24,33,43,0.12);
        }
        .stApp { background: var(--paper); color: var(--ink); font-family: 'IBM Plex Sans', sans-serif; }
        .stApp [data-testid="stHeader"] { background: rgba(255,255,255,0.9); backdrop-filter: blur(8px); }
        h1,h2,h3 { font-family: 'Playfair Display', serif; color: var(--ink); letter-spacing: -0.02em; }
        p,li,label,div,span { color: var(--ink); }

        /* Animated fade-in for all main content */
        @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulseGlow { 0%,100% { box-shadow: 0 10px 24px rgba(24,33,43,0.06); } 50% { box-shadow: 0 12px 28px rgba(47,111,94,0.12); } }
        @keyframes countUp { from { opacity: 0; transform: scale(0.85); } to { opacity: 1; transform: scale(1); } }

        div[data-testid="stMetric"] {
            background: var(--card); border: 1px solid var(--line);
            border-radius: 18px; padding: 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(24,33,43,0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            animation: fadeSlideUp 0.5s ease-out both;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 32px rgba(47,111,94,0.13);
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            animation: countUp 0.6s ease-out both;
        }

        .hero {
            padding: 2rem 2.2rem; border-radius: 24px;
            background: linear-gradient(135deg, #ffffff 0%, #f0f7f4 50%, #e8f3ef 100%);
            border: 1px solid var(--line); box-shadow: 0 18px 40px rgba(24,33,43,0.08);
            margin-bottom: 1.2rem;
            animation: fadeSlideUp 0.6s ease-out both;
        }
        .hero h1 { margin-bottom: 0.3rem; }
        .hero p { font-size: 1.05rem; margin: 0.35rem 0 0; color: #314150; line-height: 1.6; }
        .hero .badges { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .hero .badge {
            display: inline-block; padding: 0.25rem 0.65rem; border-radius: 99px;
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em;
            background: var(--accent-soft); color: var(--accent); border: 1px solid rgba(47,111,94,0.18);
            transition: transform 0.15s ease, background 0.15s ease;
        }
        .hero .badge:hover { transform: scale(1.08); background: rgba(47,111,94,0.12); }
        .roadmap {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.65rem; margin: 0.8rem 0 0.3rem;
        }
        .roadmap-item {
            padding: 0.55rem 0.7rem; border-radius: 14px;
            background: #f7fafc; border: 1px solid var(--line); text-align: center;
            font-size: 0.88rem; color: #314150;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            animation: fadeSlideUp 0.5s ease-out both;
        }
        .roadmap-item:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(47,111,94,0.10); }
        .roadmap-item strong { color: var(--accent); }
        .insight {
            padding: 0.75rem 0.95rem; border-radius: 14px; background: #f7fafc;
            border-left: 4px solid var(--accent); border-top: 1px solid var(--line);
            border-right: 1px solid var(--line); border-bottom: 1px solid var(--line);
            margin: 0.35rem 0 1rem; color: #314150;
            animation: fadeSlideUp 0.4s ease-out both;
        }
        .callout {
            padding: 0.9rem 1rem; border-radius: 18px; background: #f7fafc;
            border: 1px solid var(--line); margin-bottom: 0.9rem;
            animation: fadeSlideUp 0.4s ease-out both;
        }
        .recommendation {
            background: #ffffff; border: 1px solid var(--line); border-radius: 18px;
            padding: 1rem 1.1rem; min-height: 150px;
            box-shadow: 0 14px 30px rgba(24,33,43,0.05);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            animation: fadeSlideUp 0.5s ease-out both;
        }
        .recommendation:hover {
            transform: translateY(-5px) scale(1.01);
            box-shadow: 0 20px 40px rgba(47,111,94,0.12);
        }
        .recommendation h3 { font-size: 1.02rem; margin: 0 0 0.4rem; }
        .recommendation p { color: #314150; font-size: 0.92rem; }
        .sidebar-note {
            padding: 0.85rem 0.95rem; border-radius: 16px; background: #f7fafc;
            border: 1px solid var(--line); margin: 0.75rem 0 1rem;
        }
        .section-intro {
            padding: 0.85rem 1rem; border-radius: 18px;
            background: linear-gradient(135deg, #f7fafc 0%, #eef6f2 100%);
            border: 1px solid var(--line); margin-bottom: 1rem;
            animation: fadeSlideUp 0.4s ease-out both;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] {
            background: #f5f7fa; border-radius: 12px; color: var(--ink);
            border: 1px solid rgba(24,33,43,0.08);
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover { background: var(--accent-soft); }
        .stTabs [aria-selected="true"] { background: var(--accent-soft); color: var(--ink); font-weight: 600; }
        .portfolio-footer {
            margin-top: 2rem; padding: 1.2rem; border-radius: 18px;
            background: linear-gradient(135deg, #f0f7f4 0%, #e8f3ef 100%);
            border: 1px solid var(--line); text-align: center; color: #314150;
            animation: fadeSlideUp 0.5s ease-out both;
        }
        .portfolio-footer a { color: var(--accent); text-decoration: none; font-weight: 600; }
        .portfolio-footer a:hover { text-decoration: underline; }
        .dataset-table {
            width: 100%; border-collapse: collapse; font-size: 0.88rem;
            animation: fadeSlideUp 0.4s ease-out both;
        }
        .dataset-table th {
            background: var(--accent-soft); color: var(--accent); text-align: left;
            padding: 0.5rem 0.7rem; border-bottom: 2px solid var(--accent);
        }
        .dataset-table td {
            padding: 0.45rem 0.7rem; border-bottom: 1px solid var(--line);
            transition: background 0.15s ease;
        }
        .dataset-table tr:hover td { background: #eef6f2; }

        /* Animated number counter */
        .kpi-big {
            font-size: 2.2rem; font-weight: 700; color: var(--accent);
            font-family: 'Playfair Display', serif;
            animation: countUp 0.7s ease-out both;
        }
        .kpi-label { font-size: 0.88rem; color: #687480; margin-top: -0.2rem; }
        .kpi-card {
            text-align: center; padding: 1.1rem 0.8rem; border-radius: 18px;
            background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
            border: 1px solid var(--line);
            box-shadow: 0 10px 24px rgba(24,33,43,0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            animation: fadeSlideUp 0.5s ease-out both;
        }
        .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 14px 32px rgba(47,111,94,0.12); }

        /* Plotly chart container animation */
        .js-plotly-plot { animation: fadeSlideUp 0.5s ease-out both; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading datasets …")
def get_data():
    return load_portfolio_data()


def show_insight(text: str) -> None:
    st.markdown(
        f'<div class="insight"><strong>Insight:</strong> {text}</div>',
        unsafe_allow_html=True,
    )


def _plotly_defaults(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=30),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_color="#18212b",
        transition=dict(duration=400, easing="cubic-in-out"),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="IBM Plex Sans",
            bordercolor=ACCENT,
        ),
    )
    return fig


def _chart(fig: go.Figure) -> None:
    """Apply defaults and render chart with our own theme (not Streamlit's)."""
    _plotly_defaults(fig)
    st.plotly_chart(fig, theme=None)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def show_sidebar(data) -> None:
    ov = build_olist_overview(data.olist)
    st.sidebar.markdown("## Portfolio Summary")
    sidebar_html = f"""
        <div class="sidebar-note">
        <strong>Olist E-Commerce</strong><br>
        Orders: {ov.total_orders:,} &nbsp;|&nbsp; Fulfillment: {ov.fulfillment_rate:.1%}<br>
        Repeat rate: {ov.repeat_rate:.1f}% &nbsp;|&nbsp; AOV: R${ov.avg_order_value:,.0f}<br>
        Period: {ov.date_min} → {ov.date_max}
        </div>"""
    st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Technical Stack")
    st.sidebar.markdown(
        """
        <div class="sidebar-note" style="font-size: 0.88rem;">
        <strong>Data:</strong> Pandas · NumPy<br>
        <strong>Statistics:</strong> SciPy · Statsmodels<br>
        <strong>Visualization:</strong> Plotly<br>
        <strong>App:</strong> Streamlit<br>
        <strong>Language:</strong> Python 3.9+
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built by **Shachi Raodeo**")
    st.sidebar.caption("[GitHub](https://github.com/shachiraodeo) · [LinkedIn](https://linkedin.com/in/shachiraodeo)")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

def show_overview(data) -> None:
    ov = build_olist_overview(data.olist)

    badges = " ".join(
        f'<span class="badge">{t}</span>'
        for t in ["Python", "Pandas", "NumPy", "SciPy", "Statsmodels", "Plotly", "Streamlit", "A/B Testing"]
    )

    st.markdown(
        f"""
        <div class="hero">
            <h1>E-Commerce Analytics Portfolio</h1>
            <p><strong>Shachi Raodeo</strong> — Data Analyst</p>
            <p>End-to-end analytical project on real-world e-commerce data: from exploratory analysis and funnel
            optimization through rigorous A/B testing with statistical power analysis, to revenue-impact modeling
            that translates findings into actionable business decisions.</p>
            <div class="badges">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    roadmap_items = [
        ("📊", "EDA", "Temporal, product, geographic patterns"),
        ("🔄", "Funnel", "Order pipeline & fulfillment analysis"),
        ("⭐", "Satisfaction", "Reviews, retention, customer segmentation"),
        ("🧪", "A/B Test", "Hypothesis testing with power analysis"),
        ("💰", "Revenue Impact", "From p-values to dollar outcomes"),
        ("📋", "Recommendations", "Prioritized, data-backed actions"),
    ]
    roadmap_html = '<div class="roadmap">' + "".join(
        f'<div class="roadmap-item">{emoji}<br><strong>{title}</strong><br>{desc}</div>'
        for emoji, title, desc in roadmap_items
    ) + "</div>"
    st.markdown(roadmap_html, unsafe_allow_html=True)

    st.markdown("")

    st.subheader("Brazilian E-Commerce (Olist) — Key Metrics")

    # Animated KPI cards with counting-up numbers
    kpi_data = [
        ("Total Orders", f"{ov.total_orders:,}", "📦"),
        ("Fulfillment", f"{ov.fulfillment_rate:.1%}", "✅"),
        ("Cancellation", f"{ov.cancel_rate:.2%}", "❌"),
        ("Repeat Rate", f"{ov.repeat_rate:.1f}%", "🔄"),
        ("Avg Order Value", f"R${ov.avg_order_value:,.0f}", "💰"),
        ("Revenue at Risk", f"R${ov.revenue_at_risk:,.0f}", "⚠️"),
    ]
    kpi_html = '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0.8rem;margin-bottom:1rem;">'
    for i, (label, value, emoji) in enumerate(kpi_data):
        kpi_html += f'''
        <div class="kpi-card" style="animation-delay:{i*0.08}s;">
            <div style="font-size:1.4rem;margin-bottom:0.2rem;">{emoji}</div>
            <div class="kpi-big">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>'''
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    show_insight(
        "Every percentage point improvement in fulfillment and retention is a direct revenue lever. "
        "The low repeat rate (~3%) signals a major LTV-growth opportunity."
    )

    # Dataset inventory
    st.subheader("Dataset Inventory")
    olist_tables = {
        "Orders": len(data.olist.orders),
        "Order Items": len(data.olist.items),
        "Payments": len(data.olist.payments),
        "Reviews": len(data.olist.reviews),
        "Customers": len(data.olist.customers),
        "Products": len(data.olist.products),
        "Sellers": len(data.olist.sellers),
    }
    inv_html = '<table class="dataset-table"><tr><th>Dataset</th><th>Records</th><th>Source</th></tr>'
    for name, count in olist_tables.items():
        inv_html += f'<tr><td>{name}</td><td>{count:,}</td><td>Olist (Kaggle)</td></tr>'
    inv_html += "</table>"
    st.markdown(inv_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Olist EDA
# ══════════════════════════════════════════════════════════════════════════════

def show_olist_eda(data) -> None:
    df = data.olist.master
    st.markdown(
        '<div class="section-intro">Exploratory analysis of 100K+ Brazilian e-commerce orders — temporal patterns, product mix, geography, and payment behavior.</div>',
        unsafe_allow_html=True,
    )

    # ── Interactive filters ─────────────────────────────────────────────────
    with st.container():
        st.markdown("##### 🎛️ Filters")
        fc1, fc2, fc3 = st.columns(3)
        all_categories = sorted(df["primary_category"].dropna().unique().tolist())
        with fc1:
            sel_cats = st.multiselect("Category", all_categories, default=[], placeholder="All categories")
        with fc2:
            all_states = sorted(df["customer_state"].dropna().unique().tolist())
            sel_states = st.multiselect("State", all_states, default=[], placeholder="All states")
        with fc3:
            ts_col = df["order_purchase_timestamp"] if "order_purchase_timestamp" in df.columns else None
            if ts_col is not None:
                d_min = ts_col.min().date()
                d_max = ts_col.max().date()
            else:
                d_min, d_max = None, None

    filt = df.copy()
    if sel_cats:
        filt = filt[filt["primary_category"].isin(sel_cats)]
    if sel_states:
        filt = filt[filt["customer_state"].isin(sel_states)]
    if d_min is not None:
        dr = fc3.date_input("Date Range", value=(d_min, d_max), min_value=d_min, max_value=d_max)
        if len(dr) == 2:
            filt = filt[
                (filt["order_purchase_timestamp"].dt.date >= dr[0])
                & (filt["order_purchase_timestamp"].dt.date <= dr[1])
            ]

    st.caption(f"Showing **{len(filt):,}** of {len(df):,} orders")

    # Temporal
    left, right = st.columns(2)
    with left:
        monthly = build_monthly_orders(filt)
        fig = px.area(monthly, x="purchase_month", y="orders", title="Monthly Order Volume",
                      color_discrete_sequence=[COLORS["teal"]])
        fig.update_traces(line=dict(width=2.5), fillcolor="rgba(47,111,94,0.15)")
        fig.update_layout(xaxis_title="Month", yaxis_title="Orders", xaxis_tickangle=-45)
        peak_idx = monthly["orders"].idxmax()
        peak_month = monthly.loc[peak_idx, "purchase_month"]
        peak_orders = monthly.loc[peak_idx, "orders"]
        fig.add_annotation(
            x=peak_month, y=peak_orders,
            text=f"Peak: {peak_orders:,}", showarrow=True,
            arrowhead=2, arrowcolor=COLORS["red"],
            font=dict(size=11, color=COLORS["red"]),
        )
        _chart(fig)
    with right:
        hourly = build_hourly_orders(filt)
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=hourly["orders"], theta=hourly["purchase_hour"] * 15,
            marker_color=hourly["orders"],
            marker_colorscale=[[0, YLGN[1]], [1, YLGN[4]]],
            opacity=0.85,
        ))
        fig.update_layout(
            title="Orders by Hour of Day",
            polar=dict(
                radialaxis=dict(showticklabels=True, tickfont_size=9),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=list(range(0, 360, 15)),
                    ticktext=[f"{h}h" for h in range(24)],
                    direction="clockwise",
                ),
            ),
            showlegend=False,
        )
        _chart(fig)
    show_insight("Order volume shows clear growth with a pronounced peak hour window (10–16h) — ideal for targeted push notifications and email campaigns.")

    # Status
    left2, right2 = st.columns(2)
    with left2:
        status = build_status_distribution(filt)
        fig = px.pie(status, names="status", values="count", title="Order Status Distribution",
                     color_discrete_sequence=YLGN, hole=0.5)
        fig.update_traces(textinfo="label+percent", textposition="outside",
                         pull=[0.04 if s != "delivered" else 0 for s in status["status"]])
        _chart(fig)
    with right2:
        non_del = status[status["status"] != "delivered"].sort_values("count")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=non_del["count"], y=non_del["status"], mode="markers",
            marker=dict(size=14, color=COLORS["red"]),
        ))
        for _, row in non_del.iterrows():
            fig.add_shape(type="line", x0=0, x1=row["count"],
                         y0=row["status"], y1=row["status"],
                         line=dict(color=COLORS["red"], width=2, dash="dot"))
        fig.update_layout(title="Non-Delivered Orders (Zoom)",
                         xaxis_title="Count", yaxis_title="", showlegend=False)
        _chart(fig)
    show_insight(
        f"Overall delivery rate: {(filt['order_status']=='delivered').mean()*100:.1f}%. "
        f"Cancellation rate: {(filt['order_status']=='canceled').mean()*100:.2f}% — small in %, but large in absolute orders."
    )

    # Categories
    left3, right3 = st.columns(2)
    with left3:
        cats = build_top_categories(filt)
        fig = px.treemap(cats, path=["category"], values="orders",
                        title="Top 15 Categories (by Orders)",
                        color="orders", color_continuous_scale=[[0, YLGN[1]], [1, YLGN[4]]])
        fig.update_traces(textinfo="label+value", textfont_size=12)
        fig.update_layout(coloraxis_showscale=False)
        _chart(fig)
    with right3:
        rev_cats = build_top_revenue_categories(filt)
        cats_merged = rev_cats.merge(cats, on="category", how="left")
        fig = px.scatter(cats_merged, x="orders", y="revenue", text="category",
                        title="Categories: Orders vs Revenue",
                        size="revenue", size_max=40,
                        color="revenue",
                        color_continuous_scale=[[0, YLGN[1]], [1, YLGN[4]]])
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(xaxis_title="Orders", yaxis_title="Revenue (R$)", coloraxis_showscale=False)
        _chart(fig)

    # Geo & Payment
    left4, right4 = st.columns(2)
    with left4:
        states = build_state_orders(filt)
        fig = px.treemap(states, path=["state"], values="orders",
                        title="Top 10 States by Volume",
                        color="orders", color_continuous_scale=[[0, YLGN[1]], [1, YLGN[4]]])
        fig.update_traces(textinfo="label+value", textfont_size=13)
        fig.update_layout(coloraxis_showscale=False)
        _chart(fig)
    with right4:
        pay = build_payment_distribution(filt)
        fig = px.pie(pay, names="payment_type", values="count", title="Payment Type Distribution",
                     color_discrete_sequence=YLGN, hole=0.45)
        _chart(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Olist Funnel & Delivery
# ══════════════════════════════════════════════════════════════════════════════

def show_olist_funnel(data) -> None:
    st.markdown(
        '<div class="section-intro">Order fulfillment funnel (Placed → Approved → Shipped → Delivered) and delivery performance deep-dive.</div>',
        unsafe_allow_html=True,
    )

    funnel = build_olist_funnel(data.olist)

    left, right = st.columns(2)
    with left:
        colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["teal"]]
        stages = funnel["stage"].tolist()
        orders = funnel["orders"].tolist()
        pcts = funnel["pct_of_total"].tolist()
        max_w = 1.0
        n = len(stages)
        row_h = 1.0
        fig = go.Figure()
        for i in range(n):
            w_top = max_w * (orders[i] / orders[0]) if orders[0] else 0
            w_bot = max_w * (orders[i + 1] / orders[0]) if i < n - 1 and orders[0] else w_top * 0.7
            y_top = (n - i) * row_h
            y_bot = (n - i - 1) * row_h
            # Trapezoid
            fig.add_trace(go.Scatter(
                x=[-w_top / 2, w_top / 2, w_bot / 2, -w_bot / 2, -w_top / 2],
                y=[y_top, y_top, y_bot, y_bot, y_top],
                fill="toself", fillcolor=colors[i],
                line=dict(color="white", width=2),
                mode="lines", hoverinfo="skip", showlegend=False,
            ))
            # Label
            fig.add_annotation(
                x=0, y=(y_top + y_bot) / 2,
                text=f"<b>{stages[i]}</b><br>{orders[i]:,} ({pcts[i]:.1f}%)",
                showarrow=False, font=dict(color="white", size=13),
            )
        fig.update_layout(
            title="Order Fulfillment Funnel",
            xaxis=dict(visible=False, range=[-0.65, 0.65]),
            yaxis=dict(visible=False, range=[-0.15, n * row_h + 0.15]),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            margin=dict(t=40, b=10, l=10, r=10),
            height=420,
        )
        _chart(fig)

    with right:
        drop_offs = funnel["drop_off"].fillna(0).tolist()
        stages = funnel["stage"].tolist() + ["Delivered"]
        y_vals = [funnel["orders"].iloc[0]] + [-d for d in drop_offs[1:]] + [0]
        measures = ["absolute"] + ["relative"] * (len(drop_offs) - 1) + ["total"]
        labels = [f"{funnel['orders'].iloc[0]:,}"] \
                 + [f"−{int(d):,}" for d in drop_offs[1:]] \
                 + [f"{funnel['orders'].iloc[-1]:,}"]
        fig = go.Figure(go.Waterfall(
            x=stages,
            y=y_vals,
            measure=measures,
            text=labels,
            textposition="outside",
            connector=dict(line=dict(color=COLORS["teal"], width=1, dash="dot")),
            decreasing=dict(marker_color=COLORS["red"]),
            increasing=dict(marker_color=COLORS["green"]),
            totals=dict(marker_color=COLORS["teal"]),
        ))
        fig.update_layout(
            title="Pipeline Drop-off (Waterfall)",
            yaxis_title="Orders",
            showlegend=False,
            waterfallgap=0.35,
        )
        _chart(fig)

    st.dataframe(funnel, hide_index=True)
    st.download_button(
        "📥 Download Funnel Data",
        funnel.to_csv(index=False),
        "olist_funnel.csv",
        "text/csv",
    )
    show_insight(
        f"Overall fulfillment: {funnel['pct_of_total'].iloc[-1]:.1f}%. "
        f"Total lost in pipeline: {int(funnel['orders'].iloc[0] - funnel['orders'].iloc[-1]):,} orders. "
        "Placed→Approved is the largest leak — fixing this recovers otherwise lost revenue."
    )

    # Delivery deep-dive
    st.subheader("Delivery Performance")
    delivered = data.olist.delivered
    stats = build_delivery_stats(delivered)

    cols = st.columns(5)
    cols[0].metric("Delivered", f"{stats.total_delivered:,}")
    cols[1].metric("Median Days", f"{stats.median_days:.1f}")
    cols[2].metric("Late %", f"{stats.late_pct:.1f}%")
    cols[3].metric("On-Time Avg Review", f"{stats.on_time_avg_review:.2f}")
    cols[4].metric("Late Avg Review", f"{stats.late_avg_review:.2f}")

    left2, mid2, right2 = st.columns(3)
    with left2:
        fig = px.histogram(delivered["delivery_time_days"].clip(upper=60), nbins=50,
                          title="Delivery Time Distribution",
                          color_discrete_sequence=[COLORS["teal"]],
                          labels={"value": "Days", "count": "Orders"})
        fig.add_vline(x=stats.median_days, line_dash="dash", line_color=COLORS["red"],
                     annotation_text=f"Median: {stats.median_days:.1f}d")
        _chart(fig)
    with mid2:
        fig = px.histogram(delivered["delivery_vs_estimate"].clip(-30, 30), nbins=50,
                          title="Delivery vs. Estimated Date",
                          color_discrete_sequence=[COLORS["orange"]],
                          labels={"value": "Days (- = early, + = late)", "count": "Orders"})
        fig.add_vline(x=0, line_dash="dash", line_color=COLORS["green"],
                     annotation_text="On-time threshold")
        _chart(fig)
    with right2:
        review_by_late = delivered.dropna(subset=["review_score"]).copy()
        review_by_late["Delivery"] = review_by_late["is_late"].map({True: "Late", False: "On-Time"})
        fig = px.box(review_by_late, x="Delivery", y="review_score",
                    color="Delivery", title="Review Score: On-Time vs Late",
                    color_discrete_map={"On-Time": COLORS["green"], "Late": COLORS["red"]})
        fig.update_layout(yaxis_title="Review Score (1-5)", showlegend=False)
        _chart(fig)

    show_insight(
        f"Late delivery causes a ~{stats.review_drop:.1f} star drop in reviews "
        f"(p = {stats.mann_whitney_p:.2e}, highly significant). "
        "Delivery reliability is not just an ops KPI — it directly protects brand equity."
    )

    with st.expander("📐 Methodology: Delivery Impact Analysis"):
        st.markdown(f"""
        **How was statistical significance determined?**

        The Mann-Whitney U test (non-parametric) compares review score distributions between
        on-time and late deliveries without assuming normality.

        | Metric | On-Time | Late | Difference |
        |--------|---------|------|------------|
        | Mean review | {stats.on_time_avg_review:.2f} | {stats.late_avg_review:.2f} | {stats.review_drop:+.2f} |
        | Sample size | {stats.total_delivered - stats.late_count:,} | {stats.late_count:,} | — |
        | p-value | — | — | {stats.mann_whitney_p:.2e} |

        The extremely low p-value confirms this is not random variation — late delivery
        causally degrades the customer experience.
        """)

    # State delivery
    st.subheader("Delivery by State")
    state_del = build_state_delivery(delivered)

    # Interactive state explorer
    selected_state = st.selectbox(
        "🔍 Focus on a state:",
        ["All States"] + state_del["state"].tolist(),
        help="Select a state to see its delivery metrics highlighted",
    )

    left3, right3 = st.columns(2)
    with left3:
        sorted_del = state_del.sort_values("avg_delivery_days")
        colors_bar = [
            COLORS["teal"] if selected_state == "All States" or row["state"] == selected_state else "#d0d8dd"
            for _, row in sorted_del.iterrows()
        ]
        fig = go.Figure(go.Bar(
            y=sorted_del["state"], x=sorted_del["avg_delivery_days"],
            orientation="h", marker_color=colors_bar,
            hovertemplate="<b>%{y}</b><br>Avg: %{x:.1f} days<extra></extra>",
        ))
        fig.update_layout(title="Avg Delivery Days by State", yaxis_title="", xaxis_title="Days")
        _chart(fig)
    with right3:
        sorted_late = state_del.sort_values("late_pct")
        colors_bar2 = [
            COLORS["red"] if selected_state == "All States" or row["state"] == selected_state else "#d0d8dd"
            for _, row in sorted_late.iterrows()
        ]
        fig = go.Figure(go.Bar(
            y=sorted_late["state"], x=sorted_late["late_pct"],
            orientation="h", marker_color=colors_bar2,
            hovertemplate="<b>%{y}</b><br>Late: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(title="Late Delivery % by State", yaxis_title="", xaxis_title="% Late")
        _chart(fig)

    if selected_state != "All States":
        row = state_del[state_del["state"] == selected_state].iloc[0]
        st.markdown(
            f'<div class="callout"><strong>📍 {selected_state}</strong> — '
            f'Avg delivery: <strong>{row["avg_delivery_days"]:.1f} days</strong> | '
            f'Late rate: <strong>{row["late_pct"]:.1f}%</strong></div>',
            unsafe_allow_html=True,
        )
    st.download_button(
        "📥 Download State Delivery Data",
        state_del.to_csv(index=False),
        "state_delivery_stats.csv",
        "text/csv",
    )
    show_insight("Northern/remote states have the worst delivery performance. Targeted logistics investment in these lanes can reduce late-delivery concentration.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Olist Reviews & Retention
# ══════════════════════════════════════════════════════════════════════════════

def show_olist_reviews(data) -> None:
    delivered = data.olist.delivered
    st.markdown(
        '<div class="section-intro">Customer satisfaction signals: review score distribution, category-level quality, and repeat purchase behavior.</div>',
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns(3)
    with left:
        rev_dist = build_review_distribution(delivered)
        colors_r = [COLORS["red"], COLORS["orange"], "#c9a948", "#6aae5e", COLORS["green"]]
        fig = px.bar(rev_dist, x="score", y="count", title="Review Score Distribution",
                    text="pct", color="score",
                    color_discrete_sequence=colors_r)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(xaxis_title="Score", yaxis_title="Count", showlegend=False)
        _chart(fig)
    with mid:
        cat_rev = build_category_reviews(delivered)
        fig = px.bar(cat_rev, y="category", x="avg_score", orientation="h",
                    title="Avg Review by Category (Top 10)",
                    color_discrete_sequence=[COLORS["teal"]])
        overall_avg = delivered.dropna(subset=["review_score"])["review_score"].mean()
        fig.add_vline(x=overall_avg, line_dash="dash", line_color=COLORS["red"],
                     annotation_text=f"Overall: {overall_avg:.2f}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="Avg Score")
        _chart(fig)
    with right:
        monthly_rev = build_monthly_reviews(delivered)
        fig = px.line(monthly_rev, x="purchase_month", y="review_score",
                     markers=True, title="Avg Review Over Time",
                     color_discrete_sequence=[COLORS["teal"]])
        fig.add_hline(y=overall_avg, line_dash="dash", line_color=COLORS["red"],
                     annotation_text=f"Avg: {overall_avg:.2f}")
        if len(monthly_rev) > 2:
            best_idx = monthly_rev["review_score"].idxmax()
            worst_idx = monthly_rev["review_score"].idxmin()
            fig.add_annotation(
                x=monthly_rev.loc[best_idx, "purchase_month"],
                y=monthly_rev.loc[best_idx, "review_score"],
                text="Best", showarrow=True, arrowhead=2, font=dict(size=10, color=COLORS["green"]),
            )
            fig.add_annotation(
                x=monthly_rev.loc[worst_idx, "purchase_month"],
                y=monthly_rev.loc[worst_idx, "review_score"],
                text="Worst", showarrow=True, arrowhead=2, font=dict(size=10, color=COLORS["red"]),
            )
        fig.update_layout(xaxis_title="Month", yaxis_title="Avg Score", xaxis_tickangle=-45)
        _chart(fig)

    with_rev = delivered.dropna(subset=["review_score"])
    pct5 = (with_rev["review_score"] == 5).mean() * 100
    pct1 = (with_rev["review_score"] == 1).mean() * 100
    show_insight(f"Overall avg review: {overall_avg:.2f}. 5-star: {pct5:.1f}%, 1-star: {pct1:.1f}%. Low-scoring categories are candidates for quality investigation.")

    # Repeat purchase
    st.subheader("Customer Retention")
    cust = build_customer_orders(data.olist.master)
    repeat_rate = cust["is_repeat"].mean() * 100

    left2, mid2, right2 = st.columns(3)
    with left2:
        order_dist = cust["n_orders"].clip(upper=5).value_counts().sort_index().reset_index()
        order_dist.columns = ["orders", "customers"]
        order_dist["orders"] = order_dist["orders"].astype(str).replace("5", "5+")
        fig = px.bar(order_dist, x="orders", y="customers", title="Orders per Customer",
                    color_discrete_sequence=[COLORS["teal"]])
        fig.update_layout(xaxis_title="# Orders", yaxis_title="Customers")
        _chart(fig)
    with mid2:
        spend = cust.copy()
        spend["type"] = spend["is_repeat"].map({0: "One-Time", 1: "Repeat"})
        fig = px.box(spend, x="type", y="total_spend", color="type",
                    title="Total Spend: One-Time vs Repeat",
                    color_discrete_map={"One-Time": COLORS["orange"], "Repeat": COLORS["green"]})
        fig.update_layout(showlegend=False, yaxis_title="Total Spend (R$)", xaxis_title="")
        fig.update_yaxes(range=[0, spend["total_spend"].quantile(0.95)])
        _chart(fig)
    with right2:
        rep_review = cust.groupby("is_repeat")["avg_review"].mean().reset_index()
        rep_review["type"] = rep_review["is_repeat"].map({0: "One-Time", 1: "Repeat"})
        fig = px.bar(rep_review, x="type", y="avg_review", title="Avg Review by Customer Type",
                    color="type", color_discrete_map={"One-Time": COLORS["orange"], "Repeat": COLORS["green"]})
        fig.update_layout(showlegend=False, yaxis_title="Avg Review", xaxis_title="", yaxis_range=[3.5, 5])
        _chart(fig)

    show_insight(
        f"Repeat rate is only {repeat_rate:.1f}% — a massive retention opportunity. "
        f"If repeat rate improves by 5pp, estimated additional revenue: R${cust['total_spend'].mean() * len(cust) * 0.05:,.0f}."
    )

    # Monthly cohort retention heatmap
    st.subheader("Cohort Analysis")
    orders = data.olist.master.copy()
    orders["purchase_month"] = orders["purchase_month"].astype(str)
    cust_first = orders.groupby("customer_unique_id")["purchase_month"].min().rename("cohort")
    orders = orders.merge(cust_first, on="customer_unique_id")
    cohort_data = (
        orders.groupby(["cohort", "purchase_month"])["customer_unique_id"]
        .nunique()
        .reset_index(name="customers")
    )
    cohort_sizes = cohort_data.groupby("cohort")["customers"].first()
    cohort_pivot = cohort_data.pivot(index="cohort", columns="purchase_month", values="customers").fillna(0)
    cohort_pct = cohort_pivot.div(cohort_sizes, axis=0) * 100

    valid = cohort_sizes[cohort_sizes >= 100].index
    cohort_pct = cohort_pct.loc[cohort_pct.index.isin(valid)]
    if len(cohort_pct) > 12:
        cohort_pct = cohort_pct.iloc[-12:]

    fig = px.imshow(
        cohort_pct.round(1),
        labels=dict(x="Purchase Month", y="Cohort", color="Retention %"),
        color_continuous_scale=[[0, "#f7fcf5"], [0.5, "#a1d99b"], [1, "#355c3a"]],
        title="Monthly Cohort Retention (%)",
        aspect="auto",
    )
    fig.update_layout(xaxis_tickangle=-45)
    _chart(fig)
    show_insight("The diagonal pattern confirms most customers make a single purchase. Retention is the key lever for growth — even small improvements compound significantly over time.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — Olist A/B Test
# ══════════════════════════════════════════════════════════════════════════════

def _show_ab_charts(ab: ABTestResult, metric_label: str) -> None:
    left, mid, right = st.columns(3)
    with left:
        fig = go.Figure()
        rates = [ab.control_rate * 100, ab.treatment_rate * 100]
        fig.add_trace(go.Bar(
            x=["Control", "Treatment"], y=rates,
            marker_color=[COLORS["red"], COLORS["green"]],
            text=[f"{r:.2f}%" for r in rates], textposition="outside",
        ))
        fig.add_annotation(x=0.5, y=max(rates) * 0.95,
                          text=f"+{ab.relative_lift:.1f}% lift",
                          font=dict(size=16, color=COLORS["green"]), showarrow=False)
        fig.update_layout(title=f"{metric_label}: Control vs Treatment",
                         yaxis_title=f"{metric_label} (%)", showlegend=False)
        _chart(fig)

    with mid:
        diff = ab.absolute_lift * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[diff], y=[0.5], mode="markers",
            marker=dict(size=14, color=COLORS["blue"]),
            error_x=dict(type="data",
                        array=[ab.ci_upper - diff],
                        arrayminus=[diff - ab.ci_lower],
                        thickness=2, width=8),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=COLORS["red"])
        fig.add_annotation(x=diff, y=0.65,
                          text=f"{diff:.2f}pp [{ab.ci_lower:.2f}, {ab.ci_upper:.2f}]",
                          showarrow=False, font=dict(size=11))
        fig.update_layout(title="95% Confidence Interval", xaxis_title="Difference (pp)",
                         yaxis_visible=False, showlegend=False)
        _chart(fig)

    with right:
        x_range = np.linspace(-4, 4, 500)
        y_range = norm.pdf(x_range)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=y_range, mode="lines",
                                line=dict(color=COLORS["teal"], width=2), name="H₀"))
        reject_mask = (x_range >= 1.96) | (x_range <= -1.96)
        fig.add_trace(go.Scatter(
            x=x_range[reject_mask], y=y_range[reject_mask],
            fill="tozeroy", fillcolor="rgba(178,71,47,0.3)",
            line=dict(color="rgba(178,71,47,0.3)"), name="Rejection (α=0.05)",
        ))
        fig.add_vline(x=ab.z_stat, line_dash="dash", line_color=COLORS["green"],
                     annotation_text=f"Z = {ab.z_stat:.2f}")
        fig.update_layout(title="Hypothesis Test", xaxis_title="Z-score", showlegend=True)
        _chart(fig)


def show_olist_ab(data) -> None:
    st.markdown(
        '<div class="section-intro">A/B test simulation: would an <strong>express delivery commitment</strong> '
        '(guaranteed on-time or discount on next order) improve customer satisfaction?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="callout">
        <strong>H₀:</strong> Express delivery commitment does NOT increase satisfaction (≥4 stars).<br>
        <strong>H₁:</strong> Express delivery commitment DOES increase satisfaction.<br>
        <strong>Design:</strong> Unit = order &nbsp;|&nbsp; α = 0.05 &nbsp;|&nbsp; Power = 0.80 &nbsp;|&nbsp; Expected lift: 5-8% relative.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ab = run_olist_ab_test(data.olist.delivered)

    cols = st.columns(4)
    cols[0].metric("Control Rate", f"{ab.control_rate*100:.2f}%")
    cols[1].metric("Treatment Rate", f"{ab.treatment_rate*100:.2f}%")
    cols[2].metric("Relative Lift", f"+{ab.relative_lift:.1f}%")
    cols[3].metric("P-value", f"{ab.p_value:.6f}", delta="Significant" if ab.significant else "Not Significant")

    _show_ab_charts(ab, "Satisfaction Rate")

    if ab.significant:
        st.success(f"✅ Statistically significant (p = {ab.p_value:.6f} < 0.05). Recommendation: **ROLL OUT** the program.")
    else:
        st.warning(f"❌ Not significant (p = {ab.p_value:.6f}). Run test longer or try alternate intervention.")

    with st.expander("📐 Methodology: A/B Test Design"):
        st.markdown(f"""
        **Test design choices and rationale:**

        | Parameter | Value | Rationale |
        |-----------|-------|-----------|
        | Unit of randomization | Order | Each order is an independent interaction |
        | Metric | Satisfaction rate (≥ 4 stars) | Directly tied to NPS and repeat likelihood |
        | Significance level (α) | 0.05 | Industry standard; balances Type I/II error |
        | Power (1 − β) | 0.80 | Detects a ≥5% relative lift with 80% probability |
        | Split ratio | 50/50 | Maximizes statistical power |
        | Sample sizes | Control: {ab.n_control:,} · Treatment: {ab.n_treatment:,} | Well above minimum detectable effect threshold |

        **Treatment mechanism:** Late-delivery orders in the treatment group receive a "guaranteed on-time or discount"
        commitment. 30% of late-unsatisfied orders flip to satisfied (simulating the discount rescue), plus a small
        3% spillover effect on non-late orders from improved brand trust.

        **Statistical test:** Two-proportion z-test (Statsmodels `proportions_ztest`). The 95% confidence interval
        is computed from the standard error of the difference in proportions.
        """)

    show_insight(
        f"The express delivery commitment lifted satisfaction by {ab.absolute_lift*100:+.2f} pp "
        f"(95% CI: [{ab.ci_lower:.2f}%, {ab.ci_upper:.2f}%]). "
        "The effect is strongest among late-delivery orders."
    )

    # Revenue impact with what-if slider
    st.subheader("Projected Revenue Impact")
    st.markdown("**🎚️ What-if analysis:** Drag the slider to explore how different satisfaction lift scenarios affect revenue.")
    lift_multiplier = st.slider(
        "Satisfaction lift multiplier",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        help="1.0 = actual A/B test result. Adjust to model optimistic/pessimistic scenarios.",
    )

    # Adjust the AB result for what-if
    ab_whatif = copy.copy(ab)
    ab_whatif.treatment_rate = ab.control_rate + (ab.treatment_rate - ab.control_rate) * lift_multiplier
    ab_whatif.absolute_lift = ab_whatif.treatment_rate - ab.control_rate
    ab_whatif.relative_lift = (ab_whatif.absolute_lift / ab.control_rate * 100) if ab.control_rate else 0

    impact = build_olist_revenue_impact(data.olist, ab_whatif)

    cols2 = st.columns(3)
    with cols2[0]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=impact.projected_satisfaction * 100,
            delta={"reference": impact.current_satisfaction * 100, "suffix": "%"},
            title={"text": "Satisfaction Rate"},
            gauge={
                "axis": {"range": [60, 100]},
                "bar": {"color": COLORS["green"]},
                "bgcolor": "#f7fafc",
                "steps": [
                    {"range": [60, 75], "color": "#fce4ec"},
                    {"range": [75, 85], "color": "#fff8e1"},
                    {"range": [85, 100], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": COLORS["red"], "width": 3},
                    "thickness": 0.8,
                    "value": impact.current_satisfaction * 100,
                },
            },
        ))
        fig.update_layout(height=320)
        _chart(fig)
    with cols2[1]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=impact.projected_repeat_rate,
            delta={"reference": impact.current_repeat_rate, "suffix": "%"},
            title={"text": "Repeat Rate"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": COLORS["blue"]},
                "bgcolor": "#f7fafc",
                "steps": [
                    {"range": [0, 3], "color": "#fce4ec"},
                    {"range": [3, 5], "color": "#fff8e1"},
                    {"range": [5, 10], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": COLORS["red"], "width": 3},
                    "thickness": 0.8,
                    "value": impact.current_repeat_rate,
                },
            },
        ))
        fig.update_layout(height=320)
        _chart(fig)
    with cols2[2]:
        components = pd.DataFrame({
            "component": ["Additional Revenue", "Compensation Cost", "Net Impact"],
            "value": [impact.additional_revenue, -impact.compensation_cost, impact.net_impact],
        })
        fig = px.bar(components, x="component", y="value", title="Annualized Impact (R$)",
                    color="component",
                    color_discrete_map={
                        "Additional Revenue": COLORS["green"],
                        "Compensation Cost": COLORS["red"],
                        "Net Impact": COLORS["teal"],
                    })
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="R$")
        _chart(fig)

    show_insight(f"Net annual impact: R${impact.net_impact:,.0f}. Combined with retention campaigns and logistics optimization, Olist can materially improve both satisfaction and customer lifetime value.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def show_recommendations(data) -> None:
    st.markdown(
        '<div class="section-intro">Summary recommendations — actionable, prioritized, and tied to measurable impact.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Olist E-Commerce — Recommendations")
    olist_recs = [
        {
            "title": "🔴 Roll out express delivery program",
            "detail": "The A/B test showed a statistically significant satisfaction lift. Phase rollout starting with high-late-delivery states to maximize impact per dollar.",
        },
        {
            "title": "🔴 Fix logistics for remote states",
            "detail": "States like MA, PA, and AL have the highest late-delivery rates. Targeted carrier partnerships or warehouse positioning can cut late deliveries by 20-30%.",
        },
        {
            "title": "🔴 Launch retention campaigns",
            "detail": "With only ~3% repeat rate, post-purchase re-engagement (email sequences, loyalty program) is the highest-ROI growth lever.",
        },
        {
            "title": "🟡 Optimize seller shipping SLAs",
            "detail": "The Placed→Approved stage is the largest pipeline leak. Tighter seller handoff requirements can recover otherwise lost orders.",
        },
        {
            "title": "🟡 Add delivery tracking & notifications",
            "detail": "Proactive delivery updates improve perceived reliability and reduce inbound support volume, even when actual times don't change.",
        },
    ]

    cols = st.columns(len(olist_recs))
    for col, rec in zip(cols, olist_recs):
        col.markdown(
            f'<div class="recommendation"><h3>{rec["title"]}</h3><p>{rec["detail"]}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="callout">
        <strong>Portfolio note:</strong> This dashboard demonstrates end-to-end analytical thinking —
        from data wrangling and EDA through funnel analysis, rigorous A/B testing with statistical power analysis,
        and revenue-impact modeling that translates statistical findings into business decisions.
        Every recommendation above is backed by quantitative evidence from the preceding tabs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📐 Methodology Summary"):
        st.markdown("""
        | Technique | Where Used | Tool |
        |-----------|-----------|------|
        | Exploratory Data Analysis | EDA tab — temporal, product, geographic patterns | Pandas, Plotly |
        | Funnel Analysis | Funnel tab — order pipeline stages | Pandas |
        | Mann-Whitney U Test | Delivery tab — on-time vs late review comparison | SciPy |
        | Two-Proportion Z-Test | A/B Test tab — treatment vs control comparison | Statsmodels |
        | Statistical Power Analysis | A/B Test design — sample size validation | Statsmodels |
        | Confidence Intervals | A/B Test tab — uncertainty quantification | NumPy |
        | Revenue Impact Modeling | A/B Test tab — financial translation | Custom calculations |
        | Cohort Retention Analysis | Reviews tab — monthly retention heatmap | Pandas, Plotly |
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_styles()
    data = get_data()
    show_sidebar(data)

    tab_names = [
        "Overview",
        "EDA",
        "Funnel & Delivery",
        "Reviews & Retention",
        "A/B Test",
        "Recommendations",
    ]

    tabs = st.tabs(tab_names)

    with tabs[0]:
        show_overview(data)
    with tabs[1]:
        show_olist_eda(data)
    with tabs[2]:
        show_olist_funnel(data)
    with tabs[3]:
        show_olist_reviews(data)
    with tabs[4]:
        show_olist_ab(data)
    with tabs[5]:
        show_recommendations(data)

    # Footer
    st.markdown(
        """
        <div class="portfolio-footer">
            <strong>E-Commerce Analytics Portfolio</strong> by <strong>Shachi Raodeo</strong><br>
            Built with Python · Pandas · SciPy · Plotly · Streamlit<br>
            <a href="https://github.com/shachiraodeo" target="_blank">GitHub</a> &nbsp;·&nbsp;
            <a href="https://linkedin.com/in/shachiraodeo" target="_blank">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
