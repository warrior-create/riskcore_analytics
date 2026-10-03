"""
MarketRisk-Lab: Streamlit Dashboard

Tabs
----
1. Portfolio      – positions, market values, Greeks
2. Risk           – VaR/ES by model and confidence level, time series
3. Backtesting    – traffic light, Kupiec, Christoffersen results
4. Exceptions     – exception log, classification analysis
5. Stress         – scenario P&L, reverse stress
6. Governance     – model inventory, KRI monitoring
7. What-If Trade  – incremental VaR for proposed trades

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RiskCore - Enterprise Portfolio Risk",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS overrides for premium look
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0F172A !important;
}

/* Hide default Streamlit branding and sidebar toggle */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="collapsedControl"] {display: none;}

.main { background: #0F172A; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px; }

.stApp {
    background: #0F172A;
    color: #F8FAFC;
}

.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2);
}

.status-green { background: #1E293B; border-left: 4px solid #34D399; color: #A7F3D0; padding: 10px; border-radius: 4px; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155;}
.status-amber { background: #1E293B; border-left: 4px solid #FBBF24; color: #FDE68A; padding: 10px; border-radius: 4px; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155;}
.status-red   { background: #1E293B; border-left: 4px solid #F87171; color: #FECACA; padding: 10px; border-radius: 4px; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155;}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #334155;
}

div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; color: #F8FAFC; }
div[data-testid="stMetricLabel"] { font-size: 0.875rem; color: #94A3B8; font-weight: 500; }

.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 0;
    color: #94A3B8;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 10px 0px;
    letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #818CF8 !important;
    border-bottom: 3px solid #818CF8 !important;
}

/* Minimalist Inputs */
.stSelectbox div[data-baseweb="select"], .stTextInput input {
    background-color: #1E293B;
    border: 1px solid #475569;
    border-radius: 6px;
    color: #F8FAFC;
}
.stSelectbox div[data-baseweb="select"] * {
    color: #F8FAFC !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load config & data
# ---------------------------------------------------------------------------

@st.cache_resource
def load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_conn():
    """Get a per-call SQLite connection (thread-safe for Streamlit)."""
    import sqlite3
    cfg = load_cfg()
    db_path = cfg["data"]["db_path"]
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_df(query: str, params=None) -> pd.DataFrame:
    try:
        conn = get_conn()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        st.warning(f"DB query failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#F8FAFC"),
)


def apply_theme(fig):
    fig.update_layout(
        **PLOTLY_THEME,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#334155", showgrid=True),
        yaxis=dict(gridcolor="#334155", showgrid=True),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# Global top header (Screener style, dark mode)
col_logo, col_space, col_login = st.columns([2, 5, 1])
with col_logo:
    st.markdown("<h2 style='margin:0; padding:0; color:#F8FAFC; font-weight:600; letter-spacing:-1px;'>Risk<span style='color:#818CF8;'>Core</span></h2>", unsafe_allow_html=True)
with col_login:
    st.button("👤 Vansh ⌄", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Settings at top instead of sidebar
cfg = load_cfg()
model_options = ["parametric", "hs", "fhs_garch", "fhs_ewma", "mc_normal", "mc_t_copula"]
conf_options = [0.95, 0.975, 0.99]

with st.expander("⚙️ Global Analytics Settings"):
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        selected_model = st.selectbox("Primary Risk Model", model_options, index=1)
    with scol2:
        selected_conf = st.selectbox("Confidence Level", conf_options, index=2)
    with scol3:
        st.write("")
        st.caption("RiskCore Engine v1.0")

if "prev_model" not in st.session_state:
    st.session_state["prev_model"] = selected_model
    st.session_state["prev_conf"] = selected_conf

if selected_model != st.session_state["prev_model"] or selected_conf != st.session_state["prev_conf"]:
    st.toast(f"Risk analytics updated for {selected_model.upper()} at {selected_conf} confidence.", icon="⚡")
    st.session_state["prev_model"] = selected_model
    st.session_state["prev_conf"] = selected_conf
    st.markdown("""
        <style>
        div[data-testid="stToast"] {
            animation: fadeOut 0.5s ease 1.5s forwards;
        }
        @keyframes fadeOut {
            to { opacity: 0; display: none; }
        }
        </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "Portfolio",
    "Risk Analytics",
    "Backtesting",
    "Exception Log",
    "Stress Testing",
    "Governance",
    "Margin & Capital",
    "What-If Analysis",
    "P&L Explain",
    "Reporting"
])


# ============================================================
# TAB 1: PORTFOLIO
# ============================================================

with tabs[0]:
    st.markdown("<h1 style='text-align: center; color: #F8FAFC; margin-bottom: 5px; font-weight: 600;'>RiskCore</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 40px; font-size: 1.1rem;'>Enterprise Portfolio Risk & Analytics Platform in India.</p>", unsafe_allow_html=True)
    
    # Portfolio Upload Feature
    col_u1, col_u2, col_u3 = st.columns([1, 2, 1])
    with col_u2:
        uploaded_file = st.file_uploader("Upload custom portfolio (CSV with 'ticker', 'quantity' columns)", type=["csv"])
        
    cfg = load_cfg()
    port_cfg = cfg["portfolio"]
    equities = port_cfg["equities"].copy()  # Default from config

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            # Normalize column names
            df_upload.columns = [c.strip().lower() for c in df_upload.columns]
            if "ticker" in df_upload.columns and "quantity" in df_upload.columns:
                equities = dict(zip(df_upload["ticker"], df_upload["quantity"]))
                st.success("Custom portfolio loaded and analyzed!")
            else:
                st.error("CSV must contain 'ticker' and 'quantity' columns.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Active Portfolio Overview</div>', unsafe_allow_html=True)

    # Load latest prices
    tickers = list(equities.keys()) + [cfg["data"]["nifty_ticker"]]
    tickers_sql = ",".join(f"'{t}'" for t in tickers)
    prices_df = load_df(
        f"""SELECT ticker, MAX(date) as date, adj_close
            FROM prices WHERE ticker IN ({tickers_sql})
            GROUP BY ticker"""
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_pos = len(equities) + len(port_cfg.get("options", {})) + 2
        st.metric("Total Positions", n_pos)
    with col2:
        if not prices_df.empty:
            mv = sum(
                equities.get(row["ticker"], 0) * row["adj_close"]
                for _, row in prices_df.iterrows()
                if row["ticker"] in equities
            )
            st.metric("Equity MV (INR)", f"₹{mv:,.0f}")
        else:
            st.metric("Equity MV (INR)", "N/A")
    with col3:
        latest_vix = load_df("SELECT value FROM vol_index WHERE index_name='INDIAVIX' ORDER BY date DESC LIMIT 1")
        vix_val = f"{float(latest_vix.iloc[0,0]):.2f}" if not latest_vix.empty else "N/A"
        st.metric("India VIX", vix_val)
    with col4:
        fx_rate = load_df("SELECT rate FROM fx_rates WHERE ccy_pair='USDINR' ORDER BY date DESC LIMIT 1")
        fx_val = f"₹{float(fx_rate.iloc[0,0]):.2f}" if not fx_rate.empty else "N/A"
        st.metric("USDINR Spot", fx_val)

    st.markdown("<br>", unsafe_allow_html=True)

    # Equity positions table
    st.subheader("Holdings")
    if not prices_df.empty:
        pos_data = []
        for ticker, qty in equities.items():
            price_row = prices_df[prices_df["ticker"] == ticker]
            price = float(price_row.iloc[0]["adj_close"]) if not price_row.empty else 0
            pos_data.append({
                "Ticker": ticker,
                "Quantity": qty,
                "Last Price (INR)": f"₹{price:,.2f}",
                "Market Value (INR)": f"₹{qty*price:,.0f}",
                "Weight (%)": f"{qty*price/(sum(equities[t]*p for t in equities for _, row in prices_df[prices_df['ticker']==t].iterrows() for p in [row['adj_close']]) + 1e-6)*100:.1f}%" if qty > 0 else "0%",
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

    # Price history chart
    st.subheader("Price History (Normalised)")
    hist_df = load_df(
        f"""SELECT date, ticker, adj_close FROM prices
            WHERE ticker IN ({tickers_sql})
            ORDER BY date"""
    )
    if not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        pivot = hist_df.pivot(index="date", columns="ticker", values="adj_close")
        normed = (pivot / pivot.iloc[0] * 100).reset_index().melt(id_vars="date", var_name="Ticker", value_name="Normalised Price")
        fig = px.line(normed, x="date", y="Normalised Price", color="Ticker",
                      title="Normalised Price History (Base=100)")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2: RISK
# ============================================================

with tabs[1]:
    st.markdown('<div class="section-title">VaR / ES Summary</div>', unsafe_allow_html=True)

    var_df = load_df(
        """SELECT date, model, confidence, var_1d, es_1d
           FROM var_results
           ORDER BY date DESC, model, confidence"""
    )

    if var_df.empty:
        st.info("No VaR results found. Run `marketrisk backtest` first.")
    else:
        var_df["date"] = pd.to_datetime(var_df["date"])

        # Latest values
        latest = var_df.groupby(["model", "confidence"]).first().reset_index()
        st.subheader("Latest VaR and ES by Model")

        cols = st.columns(3)
        key_models = ["hs", "fhs_garch", "mc_t_copula"]
        for i, model in enumerate(key_models):
            row = latest[(latest["model"] == model) & (latest["confidence"] == selected_conf)]
            with cols[i]:
                if not row.empty:
                    var_val = float(row.iloc[0]["var_1d"])
                    es_val = float(row.iloc[0]["es_1d"]) if pd.notna(row.iloc[0]["es_1d"]) else var_val * 1.2
                    st.metric(f"{model.upper()} VaR ({selected_conf*100:.0f}%)", f"{var_val:.4f}")
                    st.metric(f"{model.upper()} ES ({selected_conf*100:.0f}%)", f"{es_val:.4f}")

        st.divider()
        st.subheader("VaR Time Series")

        model_var = var_df[
            (var_df["model"] == selected_model) & (var_df["confidence"] == selected_conf)
        ].sort_values("date")

        if not model_var.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=model_var["date"], y=model_var["var_1d"],
                name=f"VaR {selected_conf*100:.0f}%",
                line=dict(color="#60A5FA", width=1.5),
            ))
            if "es_1d" in model_var.columns:
                fig.add_trace(go.Scatter(
                    x=model_var["date"], y=model_var["es_1d"],
                    name="ES",
                    line=dict(color="#F472B6", width=1.5, dash="dash"),
                ))
            fig.update_layout(title=f"{selected_model.upper()} VaR/ES Time Series", **PLOTLY_THEME)
            st.plotly_chart(fig, use_container_width=True)

        # Model comparison bar chart
        st.subheader("Model Comparison (Latest VaR)")
        comp = latest[latest["confidence"] == selected_conf].copy()
        if not comp.empty:
            fig2 = px.bar(comp, x="model", y="var_1d", color="model",
                          title=f"VaR {selected_conf*100:.0f}% — All Models",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 3: BACKTESTING
# ============================================================

with tabs[2]:
    st.markdown('<div class="section-title">Backtesting Results</div>', unsafe_allow_html=True)

    bt_df = load_df("SELECT * FROM backtest_results ORDER BY run_date DESC LIMIT 500")

    if bt_df.empty:
        st.info("No backtesting results. Run the backtest command first.")
    else:
        # Traffic light summary
        st.subheader("Basel Traffic Light")
        tl_counts = bt_df.groupby("traffic_light").size().reset_index(name="count")
        cols = st.columns(3)
        for color, col in zip(["green", "amber", "red"], cols):
            n = int(tl_counts[tl_counts["traffic_light"] == color]["count"].sum())
            icon = {"green": "🟢", "amber": "🟡", "red": "🔴"}[color]
            with col:
                st.metric(f"{icon} {color.capitalize()}", n)

        st.divider()

        # Statistical test results
        st.subheader("Statistical Tests")
        display_cols = [
            "model", "confidence", "n_obs", "n_exceptions", "exception_rate",
            "traffic_light", "kupiec_pvalue", "chr_cc_pvalue"
        ]
        available = [c for c in display_cols if c in bt_df.columns]
        styled = bt_df[available].head(30)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Exception rate time series
        st.subheader("Exception Rate by Model")
        exc_df = load_df(
            """SELECT model, confidence, exception_rate, test_end as date
               FROM backtest_results ORDER BY test_end"""
        )
        if not exc_df.empty:
            fig = px.line(exc_df, x="date", y="exception_rate", color="model",
                          facet_col="confidence", markers=True,
                          title="Rolling Exception Rate")
            fig.update_xaxes(type='category', title_text='')
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 4: EXCEPTIONS
# ============================================================

with tabs[3]:
    st.markdown('<div class="section-title">Exception Log</div>', unsafe_allow_html=True)

    exc_df = load_df(
        """SELECT date, model, confidence, var_estimate, actual_loss,
                  excess_loss, classification, narrative
           FROM exception_log ORDER BY date DESC"""
    )

    if exc_df.empty:
        st.info("No exceptions recorded.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Exceptions", len(exc_df))
        with col2:
            st.metric("Unique Classification Types", exc_df["classification"].nunique())
        with col3:
            max_excess = float(exc_df["excess_loss"].max()) if not exc_df.empty else 0
            st.metric("Max Excess Loss", f"{max_excess:.4f}")

        st.divider()

        # Classification pie chart
        fig = px.pie(
            exc_df.groupby("classification").size().reset_index(name="count"),
            values="count", names="classification",
            title="Exception Classification",
            color_discrete_sequence=px.colors.sequential.Plasma,
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Timeline
        exc_df["date"] = pd.to_datetime(exc_df["date"])
        fig2 = px.scatter(exc_df, x="date", y="excess_loss", color="classification",
                          title="Exception Timeline by Classification",
                          symbol="model", size="excess_loss",
                          color_discrete_sequence=px.colors.qualitative.Bold)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        # Table
        st.subheader("Exception Details")
        selected_class = st.multiselect(
            "Filter by classification",
            options=exc_df["classification"].unique().tolist(),
            default=exc_df["classification"].unique().tolist(),
        )
        filtered = exc_df[exc_df["classification"].isin(selected_class)]
        st.dataframe(filtered.drop(columns=["narrative"], errors="ignore"),
                     use_container_width=True, hide_index=True)
        if st.checkbox("Show narratives"):
            for _, row in filtered.iterrows():
                with st.expander(f"{row['date'].date()} | {row['model']} | {row['classification']}"):
                    st.write(row.get("narrative", "No narrative"))


# ============================================================
# TAB 5: STRESS
# ============================================================

with tabs[4]:
    st.markdown('<div class="section-title">Stress Testing</div>', unsafe_allow_html=True)

    st.subheader("Historical Scenarios")

    scenarios = {
        "2008 GFC": {"Nifty Shock": -0.55, "VIX Spike": +200, "USDINR Move": +0.20},
        "2013 Taper Tantrum": {"Nifty Shock": -0.18, "VIX Spike": +80, "USDINR Move": +0.12},
        "2016 Demonetisation": {"Nifty Shock": -0.10, "VIX Spike": +60, "USDINR Move": +0.02},
        "2018 IL&FS": {"Nifty Shock": -0.12, "VIX Spike": +70, "USDINR Move": +0.08},
        "2020 COVID": {"Nifty Shock": -0.38, "VIX Spike": +350, "USDINR Move": +0.07},
        "2022 Rate Hikes": {"Nifty Shock": -0.16, "VIX Spike": +50, "USDINR Move": +0.05},
    }

    sc_df = pd.DataFrame(scenarios).T
    sc_df.index.name = "Scenario"
    st.dataframe(sc_df.style.format({
        "Nifty Shock": "{:.0%}",
        "VIX Spike": "+{:.0f}%",
        "USDINR Move": "+{:.0%}",
    }), use_container_width=True)

    st.divider()
    st.subheader("Hypothetical Scenario Builder")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        nifty_shock = st.slider("Nifty Shock (%)", -50, 20, -20)
    with col2:
        usdinr_shock = st.slider("USDINR Move (%)", -5, 30, 10)
    with col3:
        rate_shock_bps = st.slider("Rate Shock (bps)", -200, 400, 200)
    with col4:
        vix_shock = st.slider("VIX Spike (abs %)", -10, 150, 50)

    if st.button("Compute Full Revaluation Stress Loss"):
        with st.spinner("Repricing portfolio..."):
            from marketrisk.pricing.pnl_explain import PnLAttributionEngine
            from marketrisk.engine.portfolio import Portfolio
            from marketrisk.data.ingest import DataIngestor
            
            cfg = load_cfg()
            ingst = DataIngestor(cfg)
            tickers = cfg["data"]["equity_tickers"] + [cfg["data"]["nifty_ticker"]]
            prices = ingst.load_prices(tickers)
            vix_df = load_df("SELECT date, value FROM vol_index WHERE index_name='INDIAVIX' ORDER BY date DESC LIMIT 1")
            fx_df = load_df("SELECT date, rate FROM fx_rates WHERE ccy_pair='USDINR' ORDER BY date DESC LIMIT 1")
            
            if len(prices) > 0 and len(vix_df) > 0 and len(fx_df) > 0:
                date_t1 = prices.index[-1]
                nifty_spot = prices.loc[date_t1, cfg["data"]["nifty_ticker"]]
                
                # Base State
                vix_spot = vix_df.iloc[0]["value"]
                fx_spot = fx_df.iloc[0]["rate"]
                r_inr = 0.065
                ytm_10y = 0.0725
                spot_equity = prices.loc[date_t1].to_dict()
                
                # Shocked State
                nifty_spot_shocked = nifty_spot * (1 + nifty_shock / 100)
                vix_spot_shocked = vix_spot + vix_shock
                fx_spot_shocked = fx_spot * (1 + usdinr_shock / 100)
                ytm_10y_shocked = ytm_10y + (rate_shock_bps / 10000)
                r_inr_shocked = r_inr + (rate_shock_bps / 10000)
                
                # Assume all equities shock identically to Nifty for simplicity in this macro stress
                spot_equity_shocked = {k: v * (1 + nifty_shock / 100) for k, v in spot_equity.items()}
                
                port = Portfolio(cfg)
                port.set_option_strikes(nifty_spot)
                
                engine = PnLAttributionEngine(port)
                attribution = engine.explain_day(
                    date="Stress",
                    spot_equity=spot_equity_shocked,
                    spot_equity_prev=spot_equity,
                    nifty_spot=nifty_spot_shocked,
                    nifty_spot_prev=nifty_spot,
                    vix=vix_spot_shocked,
                    vix_prev=vix_spot,
                    fx_spot=fx_spot_shocked,
                    fx_spot_prev=fx_spot,
                    ytm_10y=ytm_10y_shocked,
                    ytm_10y_prev=ytm_10y,
                    r_inr=r_inr_shocked,
                ).to_dict()
                
                total_stress = attribution["full_reval_pnl"]
                
                cols = st.columns(5)
                cols[0].metric("Total Stress P&L", f"₹{total_stress:,.0f}",
                               delta=f"{'SEVERE' if total_stress < -1e6 else 'MODERATE'}",
                               delta_color="inverse")
                cols[1].metric("Equity/Delta P&L", f"₹{attribution['delta_pnl']:,.0f}")
                cols[2].metric("Options Vega P&L", f"₹{attribution['vega_pnl']:,.0f}")
                cols[3].metric("FX Forward P&L", f"₹{attribution['fx_pnl']:,.0f}")
                cols[4].metric("Bonds/Rates P&L", f"₹{attribution['rates_pnl']:,.0f}")
                
            else:
                st.error("Insufficient market data in DB to run stress tests.")


# ============================================================
# TAB 6: GOVERNANCE
# ============================================================

with tabs[5]:
    st.markdown('<div class="section-title">Model Governance</div>', unsafe_allow_html=True)

    inv_df = load_df("SELECT * FROM model_inventory")

    if not inv_df.empty:
        st.subheader("Model Inventory")
        st.dataframe(inv_df, use_container_width=True, hide_index=True)
    else:
        st.info("Model inventory empty. Run the governance population command.")

    st.divider()
    st.subheader("KRI Monitoring Dashboard")
    kri_df = load_df(
        "SELECT * FROM kri_log ORDER BY date DESC LIMIT 200"
    )
    if kri_df.empty:
        st.info("No KRI data. Run the monitoring module.")
    else:
        kri_df["date"] = pd.to_datetime(kri_df["date"])
        fig = px.scatter(kri_df, x="date", y="value", color="status",
                         facet_col="kri_name", title="KRI Trends",
                         color_discrete_map={"green": "#10B981", "amber": "#F59E0B", "red": "#EF4444"})
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Open DQ Issues")
    dq_df = load_df("SELECT * FROM dq_log WHERE resolved=0 ORDER BY logged_at DESC LIMIT 50")
    if dq_df.empty:
        st.success("✓ No open data quality issues")
    else:
        st.dataframe(dq_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 7: MARGIN & CAPITAL
# ============================================================

with tabs[6]:
    st.markdown('<div class="section-title">Enterprise Capital & Margin (ERM)</div>', unsafe_allow_html=True)
    st.markdown("Integrates Basel/ISDA capital methodologies to cover Credit, Operational, and Margin requirements.")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Standardized Initial Margin (SIMM-lite)")
        from marketrisk.capital.simm_lite import portfolio_simm_im
        # Illustrative delta sensitivities from the portfolio
        eq_sens = {"NIFTY": 5_000_000, "RELIANCE": 1_200_000}
        fx_sens = {"USDINR": 4_150_000}
        ir_sens = {"INR_10Y": 800_000}
        
        simm_res = portfolio_simm_im(eq_sens, fx_sens, ir_sens)
        
        st.metric("Total Initial Margin (IM)", f"₹{simm_res['total_im']:,.0f}")
        
        simm_df = pd.DataFrame([
            {"Asset Class": "Equity", "Margin (INR)": simm_res["equity_im"]},
            {"Asset Class": "FX", "Margin (INR)": simm_res["fx_im"]},
            {"Asset Class": "Interest Rates", "Margin (INR)": simm_res["ir_im"]},
        ])
        fig_simm = px.pie(simm_df, values="Margin (INR)", names="Asset Class", hole=0.4,
                          title="SIMM Initial Margin by Asset Class",
                          color_discrete_sequence=px.colors.qualitative.Prism)
        apply_theme(fig_simm)
        st.plotly_chart(fig_simm, use_container_width=True)

    with colB:
        st.subheader("Credit Economic Capital (Vasicek)")
        from marketrisk.capital.credit_vasicek import portfolio_credit_ec
        # Illustrative counterparties
        exposures = [
            {"name": "Bank A (Swap Cpty)", "ead": 4_150_000, "pd": 0.005, "lgd": 0.45},
            {"name": "Corp B (Bond Issuer)", "ead": 1_000_000, "pd": 0.02, "lgd": 0.60},
            {"name": "Broker C (Margin)", "ead": 800_000, "pd": 0.01, "lgd": 0.50},
        ]
        credit_res = portfolio_credit_ec(exposures)
        
        st.metric("Total Credit EC (99.9%)", f"₹{credit_res['total_credit_ec']:,.0f}")
        
        credit_df = pd.DataFrame(credit_res["details"])
        fig_credit = px.bar(credit_df, x="name", y="credit_ec", title="Credit EC by Counterparty",
                            color="credit_ec", color_continuous_scale="Reds")
        apply_theme(fig_credit)
        st.plotly_chart(fig_credit, use_container_width=True)

    st.divider()
    st.subheader("Operational Risk (Poisson-Lognormal LDA)")
    from marketrisk.capital.oprisk_lda import oprisk_lda_monte_carlo
    
    col_op1, col_op2, col_op3 = st.columns(3)
    with col_op1:
        st.write("Frequency (λ events/yr): **2.5**")
    with col_op2:
        st.write("Severity (μ log): **12.0**")
    with col_op3:
        st.write("Severity (σ log): **1.5**")
        
    if st.button("Run OpRisk Monte Carlo (100k paths)"):
        with st.spinner("Simulating 100,000 loss years..."):
            op_res = oprisk_lda_monte_carlo(lambda_freq=2.5, mu_sev=12.0, sigma_sev=1.5, n_scenarios=100_000)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Loss (EL)", f"₹{op_res['expected_loss']:,.0f}")
            c2.metric("OpRisk VaR (99.9%)", f"₹{op_res['oprisk_var']:,.0f}")
            c3.metric("Economic Capital (UL)", f"₹{op_res['oprisk_ec']:,.0f}", delta="Severe Tail Risk", delta_color="inverse")
            st.info("The LDA model simulates Poisson events multiplied by Lognormal severities to find the 99.9% worst-case loss year.")

# ============================================================
# TAB 8: WHAT-IF TRADE
# ============================================================

with tabs[7]:
    st.markdown('<div class="section-title">What-If Trade Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "Add a proposed trade and see its incremental VaR, ES impact, "
        "and whether it triggers an NMRF flag."
    )

    cfg = load_cfg()
    equity_tickers = cfg["data"]["equity_tickers"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trade_ticker = st.selectbox("Ticker", equity_tickers + ["Custom"])
    with col2:
        trade_qty = st.number_input("Quantity (shares)", min_value=-10000, max_value=10000, value=100)
    with col3:
        trade_price = st.number_input("Price (INR)", min_value=0.0, value=1000.0)
    with col4:
        incr_model = st.selectbox("Model", ["hs", "parametric", "fhs_garch"], index=0)

    if st.button("Compute Incremental VaR"):
        from marketrisk.data.ingest import DataIngestor
        from marketrisk.engine.walk_forward import WalkForwardEngine

        ingst = DataIngestor(cfg)
        tickers = cfg["data"]["equity_tickers"] + [cfg["data"]["nifty_ticker"]]
        prices = ingst.load_prices(tickers)

        if prices.empty:
            st.error("No price data. Run ingest first.")
        else:
            returns = np.log(prices / prices.shift(1)).dropna()
            engine = WalkForwardEngine(
                returns=returns,
                portfolio_weights=None,
                cfg=cfg,
                conn=ingst.conn,
            )
            result = engine.incremental_var(
                trade={"ticker": trade_ticker, "quantity": trade_qty, "price": trade_price},
                model_name=incr_model,
                confidence=0.99,
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Base VaR", f"{result['base_var']:.4f}")
            col2.metric("New VaR", f"{result['new_var']:.4f}")
            col3.metric("Incremental VaR",
                        f"{result['incremental_var']:+.4f}",
                        delta_color="inverse")
            col4.metric(
                "NMRF Flag",
                "⚠️ YES" if result["is_nmrf_flagged"] else "✓ NO",
                delta_color="off",
            )
            if result["is_nmrf_flagged"]:
                st.warning(
                    f"⚠️ NMRF Flag: {trade_ticker} has only {result['n_obs']} "
                    f"observations (need 24). An additional stress charge will apply."
                )


# ============================================================
# TAB 9: P&L EXPLAIN
# ============================================================

with tabs[8]:
    st.markdown('<div class="section-title">P&L Attribution (Greeks)</div>', unsafe_allow_html=True)
    st.markdown("Decompose yesterday's portfolio P&L into Greeks (Delta, Gamma, Vega, Theta) and Unexplained residuals.")

    if st.button("Run P&L Attribution"):
        with st.spinner("Computing Greeks and P&L attribution..."):
            from marketrisk.pricing.pnl_explain import PnLAttributionEngine
            from marketrisk.data.ingest import DataIngestor
            import datetime
            
            cfg = load_cfg()
            ingst = DataIngestor(cfg)
            tickers = cfg["data"]["equity_tickers"] + [cfg["data"]["nifty_ticker"]]
            prices = ingst.load_prices(tickers)
            
            if prices.shape[0] < 2:
                st.error("Not enough data to compute daily P&L attribution.")
            else:
                # We need market state for yesterday (T-1) and today (T)
                # For demonstration, we'll construct a mock market_state series using the last 2 days
                
                vix_df = load_df("SELECT date, value FROM vol_index WHERE index_name='INDIAVIX' ORDER BY date DESC LIMIT 2")
                fx_df = load_df("SELECT date, rate FROM fx_rates WHERE ccy_pair='USDINR' ORDER BY date DESC LIMIT 2")
                
                if len(vix_df) == 2 and len(fx_df) == 2:
                    date_t = prices.index[-1]
                    date_t1 = prices.index[-2]
                    
                    from marketrisk.engine.portfolio import Portfolio
                    port = Portfolio(cfg)
                    
                    nifty_t1 = prices.loc[date_t1, cfg["data"]["nifty_ticker"]]
                    port.set_option_strikes(nifty_t1)
                    
                    engine = PnLAttributionEngine(port)
                    
                    attribution = engine.explain_day(
                        date=str(date_t.date()),
                        spot_equity=prices.loc[date_t].to_dict(),
                        spot_equity_prev=prices.loc[date_t1].to_dict(),
                        nifty_spot=prices.loc[date_t, cfg["data"]["nifty_ticker"]],
                        nifty_spot_prev=prices.loc[date_t1, cfg["data"]["nifty_ticker"]],
                        vix=vix_df.iloc[0]["value"],
                        vix_prev=vix_df.iloc[1]["value"],
                        fx_spot=fx_df.iloc[0]["rate"],
                        fx_spot_prev=fx_df.iloc[1]["rate"],
                        ytm_10y=0.0725,
                        ytm_10y_prev=0.0725,
                        r_inr=0.065,
                    ).to_dict()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Actual P&L", f"₹{attribution['full_reval_pnl']:,.0f}")
                    col2.metric("Explained P&L", f"₹{attribution['approx_pnl']:,.0f}")
                    col3.metric("Unexplained", f"₹{attribution['unexplained_pnl']:,.0f}")
                    
                    # Unexplained ratio
                    abs_actual = abs(attribution['full_reval_pnl']) + 1e-6
                    ratio = abs(attribution['unexplained_pnl']) / abs_actual
                    col4.metric("Unexplained Ratio", f"{ratio:.1%}", 
                                delta="High Error" if ratio > 0.1 else "Good Fit", 
                                delta_color="inverse" if ratio > 0.1 else "normal")
                    
                    st.divider()
                    
                    st.subheader("Attribution by Greek")
                    attr_df = pd.DataFrame([
                        {"Driver": "Delta (Directional)", "P&L (INR)": attribution["delta_pnl"]},
                        {"Driver": "Gamma (Curvature)", "P&L (INR)": attribution["gamma_pnl"]},
                        {"Driver": "Vega (Volatility)", "P&L (INR)": attribution["vega_pnl"]},
                        {"Driver": "Theta (Time Decay)", "P&L (INR)": attribution["theta_pnl"]},
                        {"Driver": "Rho (Rates)", "P&L (INR)": attribution["rates_pnl"]},
                    ])
                    
                    fig = px.bar(attr_df, x="Driver", y="P&L (INR)", title="P&L Explained by Risk Driver",
                                 color="P&L (INR)", color_continuous_scale="RdYlGn")
                    apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 10: REPORTING
# ============================================================

with tabs[9]:
    st.markdown('<div class="section-title">Automated Reporting Engine</div>', unsafe_allow_html=True)
    st.markdown("Generate and download formal regulatory PDF and Excel reports for the Risk Committee.")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("PDF Regulatory Report")
        if st.button("Generate Professional PDF Report"):
            with st.spinner("Compiling PDF report..."):
                import importlib
                import marketrisk.reporting.pdf_report
                importlib.reload(marketrisk.reporting.pdf_report)
                from marketrisk.reporting.pdf_report import PDFReporter
                import datetime
                
                target_date = datetime.date.today().strftime("%Y-%m-%d")
                cfg = load_cfg()
                if 'equities' in locals():
                    cfg["portfolio"]["equities"] = equities
                reporter = PDFReporter(cfg)
                
                try:
                    pdf_path = reporter.generate(target_date)
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.success(f"PDF generated successfully: {os.path.basename(pdf_path)}")
                    st.download_button(
                        label="Download PDF Report 📥",
                        data=pdf_bytes,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

    with col_r2:
        st.subheader("Excel Data Dump")
        if st.button("Generate Excel Report"):
            with st.spinner("Compiling Excel report..."):
                from marketrisk.reporting.excel_report import ExcelReporter
                import datetime
                
                target_date = datetime.date.today().strftime("%Y-%m-%d")
                cfg = load_cfg()
                reporter = ExcelReporter(cfg)
                
                try:
                    xl_path = reporter.generate(target_date)
                    with open(xl_path, "rb") as f:
                        xl_bytes = f.read()
                    
                    st.success(f"Excel generated successfully: {os.path.basename(xl_path)}")
                    st.download_button(
                        label="Download Excel Data 📥",
                        data=xl_bytes,
                        file_name=os.path.basename(xl_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error generating Excel: {e}")
                else:
                    st.warning("Incomplete market data to run attribution.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "RiskCore Analytics v1.0 | Enterprise Risk Management | "
    "All values illustrative. Not for regulatory submission."
)
