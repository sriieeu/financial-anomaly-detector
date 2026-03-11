"""
app.py — Financial Statement Anomaly Detector
Clean Obsidian / Amber Editorial Theme

Run with:
    streamlit run src/ui/app.py
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Financial Anomaly Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System — Obsidian / Amber
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg:           #0d0f14;
    --bg-panel:     #13161d;
    --bg-raised:    #1a1e28;
    --bg-hover:     #1e2330;
    --border:       rgba(255,255,255,.08);
    --border-mid:   rgba(255,255,255,.14);
    --text:         #e8ecf4;
    --text-mid:     rgba(232,236,244,.65);
    --text-muted:   rgba(232,236,244,.38);
    --amber:        #f59e0b;
    --amber-soft:   rgba(245,158,11,.12);
    --amber-mid:    rgba(245,158,11,.25);
    --red:          #ef4444;
    --red-soft:     rgba(239,68,68,.12);
    --orange:       #f97316;
    --orange-soft:  rgba(249,115,22,.12);
    --yellow:       #eab308;
    --yellow-soft:  rgba(234,179,8,.10);
    --green:        #22c55e;
    --green-soft:   rgba(34,197,94,.10);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text);
    background: var(--bg) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg-raised); border-radius: 99px; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main layout ── */
.main { background: var(--bg) !important; }
.main .block-container {
    padding: 0 48px 72px;
    max-width: 1280px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sidebar brand ── */
.sb-brand {
    padding: 28px 20px 22px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 6px;
}
.sb-brand-icon {
    font-size: 22px;
    margin-bottom: 8px;
}
.sb-brand-name {
    font-family: 'DM Serif Display', serif;
    font-size: 17px;
    color: var(--text) !important;
    line-height: 1.2;
    margin: 0 0 4px;
}
.sb-brand-tag {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--amber) !important;
    opacity: .85;
}

/* ── Sidebar nav (radio) ── */
section[data-testid="stSidebar"] .stRadio { padding: 6px 0; }
section[data-testid="stSidebar"] .stRadio > label { display: none !important; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 20px !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    color: var(--text-mid) !important;
    cursor: pointer !important;
    transition: background .15s, color .15s, border-color .15s !important;
    border-left: 2px solid transparent !important;
    letter-spacing: .02em !important;
    line-height: 1 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--bg-raised) !important;
    color: var(--text) !important;
    border-left-color: var(--border-mid) !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input[type="radio"]:checked) {
    background: var(--amber-soft) !important;
    color: var(--amber) !important;
    border-left-color: var(--amber) !important;
    font-weight: 500 !important;
}
/* hide the actual radio circle */
section[data-testid="stSidebar"] .stRadio label input[type="radio"] { display: none !important; }
section[data-testid="stSidebar"] .stRadio label > div:first-child { display: none !important; }

/* nav section divider label */
.sb-nav-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    padding: 14px 20px 6px;
}

/* ── Page header ── */
.page-head {
    padding: 44px 0 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
}
.page-head-kicker {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 10px;
}
.page-head h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 38px;
    font-weight: 400;
    color: var(--text);
    margin: 0 0 10px;
    line-height: 1.1;
}
.page-head h1 em {
    color: var(--amber);
    font-style: italic;
}
.page-head p {
    font-family: 'DM Sans', sans-serif;
    font-size: 13.5px;
    color: var(--text-mid);
    margin: 0;
    line-height: 1.7;
}

/* ── Section label ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 9.5px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Divider ── */
.divider { height: 1px; background: var(--border); margin: 32px 0; }

/* ── Stat tile ── */
.stat-tile {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 22px;
    transition: border-color .2s;
}
.stat-tile:hover { border-color: var(--border-mid); }
.stat-tile .num {
    font-family: 'DM Serif Display', serif;
    font-size: 34px;
    font-weight: 400;
    line-height: 1;
    margin-bottom: 5px;
}
.stat-tile .lbl {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ── Gauge ── */
.gauge-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 20px 0;
}
.gauge-ring {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    border-width: 4px;
    border-style: solid;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    background: var(--bg);
}
.gauge-num {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    line-height: 1;
}
.gauge-denom {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    color: var(--text-muted);
}
.gauge-level {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: .14em;
    text-transform: uppercase;
    font-weight: 500;
}
.gauge-caption {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
}

/* ── Risk badge ── */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 9px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 9.5px;
    font-weight: 500;
    letter-spacing: .07em;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-critical { background: var(--red-soft);    color: var(--red);    border-color: rgba(239,68,68,.25); }
.badge-high     { background: var(--orange-soft); color: var(--orange); border-color: rgba(249,115,22,.25); }
.badge-medium   { background: var(--yellow-soft); color: var(--yellow); border-color: rgba(234,179,8,.25); }
.badge-low      { background: var(--green-soft);  color: var(--green);  border-color: rgba(34,197,94,.25); }
.badge-none     { background: var(--bg-raised);   color: var(--text-muted); border-color: var(--border); }

/* ── Flag item ── */
.flag {
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 12.5px;
    line-height: 1.6;
    border-left: 3px solid;
}
.flag-critical { background: var(--red-soft);    border-color: var(--red); }
.flag-high     { background: var(--orange-soft); border-color: var(--orange); }
.flag-medium   { background: var(--yellow-soft); border-color: var(--yellow); }
.flag-low      { background: var(--green-soft);  border-color: var(--green); }
.flag strong {
    display: block;
    font-size: 10.5px;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--text);
}

/* ── Card panel ── */
.card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px;
}
.card h4 {
    font-family: 'DM Serif Display', serif;
    font-size: 16px;
    font-weight: 400;
    color: var(--text);
    margin: 0 0 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}

/* ── Info box ── */
.info-box {
    background: var(--amber-soft);
    border: 1px solid var(--amber-mid);
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'DM Sans', sans-serif;
    font-size: 12.5px;
    color: var(--text-mid);
    line-height: 1.75;
}
.info-box strong {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .08em;
    display: block;
    margin-bottom: 6px;
    color: var(--amber);
}

/* ── Streamlit component overrides ── */
.stButton > button {
    background: var(--amber) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 10px 26px !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

.stTextInput > div > div > input {
    background: var(--bg-raised) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px var(--amber-mid) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-raised) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

.stSlider > div { color: var(--text) !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--amber) !important; }

.stCheckbox label { font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; color: var(--text-mid) !important; }

.stAlert { border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; }
.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-panel) !important;
}
.stExpander summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 28px !important;
    color: var(--text) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 9.5px !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

div[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border-mid) !important;
    border-radius: 10px !important;
    background: var(--bg-raised) !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

div[data-testid="stStatus"] > div {
    background: var(--bg-panel) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    padding: 10px 16px !important;
    margin-bottom: 6px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def init():
    defaults = {
        "fin_df":        None,
        "benford_df":    None,
        "if_df":         None,
        "variance_df":   None,
        "scorecards":    None,
        "sc_df":         None,
        "tickers_used":  [],
        "analysis_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    from ingestion.edgar_client       import EDGARClient
    from ingestion.filing_downloader  import FilingDownloader
    from analysis.benfords_law        import BenfordsAnalyzer
    from analysis.variance_analyzer   import VarianceAnalyzer
    from models.isolation_forest      import AnomalyDetector
    from reporting.report_generator   import ReportGenerator

    return (
        EDGARClient(),
        FilingDownloader(),
        BenfordsAnalyzer(),
        VarianceAnalyzer(),
        AnomalyDetector(),
        ReportGenerator(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def score_color(s):
    if s >= 75: return "#ef4444"
    if s >= 50: return "#f97316"
    if s >= 25: return "#eab308"
    if s > 0:   return "#22c55e"
    return "#6b7280"

def render_gauge(score, label=""):
    color = score_color(score)
    level = ("CRITICAL" if score >= 75 else "HIGH" if score >= 50
             else "MEDIUM" if score >= 25 else "LOW" if score > 0 else "CLEAN")
    st.markdown(f"""
    <div class="gauge-wrap">
        <div class="gauge-ring" style="border-color:{color};">
            <span class="gauge-num" style="color:{color};">{score:.0f}</span>
            <span class="gauge-denom">/ 100</span>
        </div>
        <span class="gauge-level" style="color:{color};">{level}</span>
        <span class="gauge-caption">{label}</span>
    </div>""", unsafe_allow_html=True)

def section_label(text):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

def stat_tile(num, label, color="var(--text)"):
    st.markdown(f"""<div class="stat-tile">
        <div class="num" style="color:{color};">{num}</div>
        <div class="lbl">{label}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demo data generator
# ─────────────────────────────────────────────────────────────────────────────
def _make_demo_data(tickers):
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    concepts = [
        "Revenues", "NetIncomeLoss", "GrossProfit", "OperatingIncomeLoss",
        "Assets", "Liabilities", "AssetsCurrent", "LiabilitiesCurrent",
        "CashAndCashEquivalentsAtCarryingValue", "LongTermDebt",
        "StockholdersEquity", "RetainedEarningsAccumulatedDeficit",
        "OperatingExpenses", "CostOfRevenue", "ResearchAndDevelopmentExpense",
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInInvestingActivities",
        "EarningsPerShareBasic",
    ]
    base_vals = {
        "Revenues": 1e10, "NetIncomeLoss": 5e8, "GrossProfit": 3e9,
        "OperatingIncomeLoss": 2e9, "Assets": 5e10, "Liabilities": 2e10,
        "AssetsCurrent": 1e10, "LiabilitiesCurrent": 5e9,
        "CashAndCashEquivalentsAtCarryingValue": 2e9, "LongTermDebt": 8e9,
        "StockholdersEquity": 3e10, "RetainedEarningsAccumulatedDeficit": 2e10,
        "OperatingExpenses": 8e9, "CostOfRevenue": 6e9,
        "ResearchAndDevelopmentExpense": 1e9,
        "NetCashProvidedByUsedInOperatingActivities": 3e9,
        "NetCashProvidedByUsedInInvestingActivities": -1e9,
        "EarningsPerShareBasic": 5.0,
    }

    if not tickers:
        return pd.DataFrame(columns=["ticker","concept","end_date","value","unit","fiscal_year","form_type"])

    for ticker in tickers:
        scale = rng.uniform(0.5, 3.0)
        for year in range(2019, 2024):
            for concept in concepts:
                base  = base_vals.get(concept, 1e9) * scale
                noise = rng.normal(1.0, 0.12)
                trend = 1 + rng.normal(0.07, 0.05) * (year - 2019)
                val   = base * noise * trend

                if ticker == tickers[0] and year == 2022 and concept == "Revenues":
                    val *= rng.uniform(1.8, 2.5)
                if len(tickers) >= 2 and ticker == tickers[1] and year == 2021 and concept == "NetIncomeLoss":
                    val = -abs(val)

                rows.append({
                    "ticker":      ticker,
                    "concept":     concept,
                    "end_date":    pd.Timestamp(f"{year}-12-31"),
                    "value":       round(val, 2),
                    "unit":        "USD",
                    "fiscal_year": year,
                    "form_type":   "10-K",
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">◈</div>
        <div class="sb-brand-name">Financial Anomaly<br>Detector</div>
        <div class="sb-brand-tag">EDGAR · Benford · ML</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)

    nav_items = {
        "Ingest & Analyze":    "↓  Ingest & Analyze",
        "Benford's Law":       "~  Benford's Law",
        "Isolation Forest":    "⊞  Isolation Forest",
        "YoY Variance":        "±  YoY Variance",
        "Scorecard Dashboard": "□  Scorecard Dashboard",
    }
    nav = st.radio(
        "Navigation",
        list(nav_items.keys()),
        index=0,
        label_visibility="collapsed",
        format_func=lambda p: nav_items[p],
    )

    if st.session_state.analysis_done:
        tickers = st.session_state.tickers_used
        st.markdown(f"""
        <div style="margin:14px 12px 0;padding:12px 14px;background:var(--bg-raised);border:1px solid var(--border);border-radius:8px;font-family:'DM Mono',monospace;font-size:10.5px;color:var(--text-muted);line-height:1.8;">
            <div style="color:var(--amber);font-size:9px;letter-spacing:.14em;text-transform:uppercase;margin-bottom:6px;">Active dataset</div>
            {'<br>'.join([f'<span style="color:var(--text);">{t}</span>' for t in tickers])}
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Ingest & Analyze
# ─────────────────────────────────────────────────────────────────────────────
if nav == "Ingest & Analyze":
    st.markdown("""
    <div class="page-head">
        <div class="page-head-kicker">SEC EDGAR · Live Data</div>
        <h1>Financial statement<br><em>anomaly detection</em></h1>
        <p>Ingest 10‑K / 10‑Q filings, then run Benford's Law + Isolation Forest<br>
        + YoY variance analysis to surface irregularities.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        section_label("Company Selection")
        ticker_input = st.text_input(
            "Tickers",
            value="AAPL, MSFT, GOOGL, AMZN, META",
            placeholder="e.g. AAPL, MSFT, TSLA",
            label_visibility="collapsed",
            help="Comma-separated ticker symbols — all must be listed on SEC EDGAR.",
        )
        use_demo = st.checkbox(
            "Use pre-loaded demo dataset  (no API call required)",
            value=True,
        )

    with col_r:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h4>Analysis Options</h4>", unsafe_allow_html=True)
        form_type = st.selectbox("Filing type", ["10-K", "10-Q"], index=0)
        years     = st.slider("Years of history", 3, 10, 5)
        contam    = st.slider("IF contamination %", 1, 20, 5) / 100
        st.markdown("</div>", unsafe_allow_html=True)

    divider()
    run = st.button("Run Analysis →", type="primary")

    if run:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        edgar, downloader, benford, variance, detector, reporter = get_pipeline()

        try:
            with st.status("Ingesting SEC EDGAR data…", expanded=False):
                if use_demo:
                    fin_df = _make_demo_data(tickers)
                    st.write(f"✓  Demo dataset: {len(fin_df):,} rows · {len(tickers)} companies")
                else:
                    fin_df = downloader.build_dataset(tickers, form_type, years, save=False)
                    st.write(f"✓  Fetched {len(fin_df):,} rows from EDGAR")
                st.session_state.fin_df = fin_df
                st.session_state.tickers_used = tickers

            required_cols = {"ticker", "concept", "value", "fiscal_year"}
            missing_cols  = required_cols - set(fin_df.columns)

            if fin_df.empty:
                st.error("No data returned from SEC EDGAR. Try the demo dataset or different tickers.")
            elif missing_cols:
                st.error(f"Dataset missing required columns: {', '.join(sorted(missing_cols))}.")
            else:
                with st.status("Running Benford's Law analysis…", expanded=False):
                    benford_df = benford.analyze_dataframe(fin_df)
                    st.session_state.benford_df = benford_df
                    flagged = benford_df["flagged"].sum()
                    st.write(f"✓  {len(benford_df)} concept–ticker pairs · {flagged} flagged")

                with st.status("Fitting Isolation Forest…", expanded=False):
                    detector.contamination = contam
                    detector.model.set_params(contamination=contam)
                    if_df = detector.score_all(fin_df)
                    st.session_state.if_df = if_df
                    anomalies = if_df["is_anomaly"].sum()
                    st.write(f"✓  {len(if_df)} company-years scored · {anomalies} anomalies detected")

                with st.status("Running YoY variance analysis…", expanded=False):
                    variance_df = variance.flag_all(fin_df)
                    st.session_state.variance_df = variance_df
                    st.write(f"✓  {len(variance_df)} variance flags raised")

                with st.status("Building scorecards…", expanded=False):
                    company_names = {t: t for t in tickers}
                    fiscal_years  = sorted(fin_df["fiscal_year"].unique().tolist())
                    scorecards    = reporter.build_all_scorecards(
                        tickers, company_names, fiscal_years,
                        benford_df, if_df, variance_df,
                    )
                    sc_df = reporter.scorecards_to_dataframe(scorecards)
                    st.session_state.scorecards = scorecards
                    st.session_state.sc_df      = sc_df
                    high_risk = (sc_df["risk_level"].isin(["high","critical"])).sum()
                    st.write(f"✓  {len(sc_df)} scorecards · {high_risk} high/critical")

            st.session_state.analysis_done = True

            divider()
            section_label("Results Summary")

            c1, c2, c3, c4 = st.columns(4)
            with c1: stat_tile(len(tickers), "Companies", "var(--text)")
            with c2: stat_tile(int(benford_df["flagged"].sum()), "Benford Flags", "var(--red)")
            with c3: stat_tile(int(if_df["is_anomaly"].sum()), "IF Anomalies", "var(--orange)")
            with c4: stat_tile(len(variance_df), "Variance Flags", "var(--yellow)")

            st.success("✓  Analysis complete — use the sidebar to explore results.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Benford's Law
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Benford's Law":
    st.markdown("""
    <div class="page-head">
        <div class="page-head-kicker">Forensic · Digit Analysis</div>
        <h1>Benford's <em>Law</em></h1>
        <p>Digit distribution analysis across all reported financial figures.<br>
        Significant deviation from expected frequencies is a forensic indicator of manipulation.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("Run an analysis from **Ingest & Analyze** first.")
        st.stop()

    import plotly.graph_objects as go

    benford_df = st.session_state.benford_df
    fin_df     = st.session_state.fin_df
    flagged    = benford_df[benford_df["flagged"] == True]

    c1, c2, c3 = st.columns(3)
    with c1: stat_tile(len(benford_df), "Pairs Tested", "var(--text)")
    with c2: stat_tile(len(flagged), "Flagged", "var(--red)")
    with c3: stat_tile(f"{benford_df['mad'].mean():.4f}", "Avg MAD", "var(--yellow)")

    divider()

    col_l, col_r = st.columns([2, 3], gap="large")

    with col_l:
        section_label("Select Company · Concept")
        tickers_avail = sorted(benford_df["ticker"].unique())
        sel_ticker    = st.selectbox("Ticker",  tickers_avail, label_visibility="collapsed")
        concepts_avail = sorted(benford_df[benford_df["ticker"] == sel_ticker]["concept"].unique())
        sel_concept   = st.selectbox("Concept", concepts_avail, label_visibility="collapsed")

        row = benford_df[
            (benford_df["ticker"]  == sel_ticker) &
            (benford_df["concept"] == sel_concept)
        ]

        if not row.empty:
            r = row.iloc[0]
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            sc = max(0.0, 100.0 * (1.0 - float(r["mad"]) / 0.05))
            render_gauge(sc, "Benford Conformity")
            st.markdown(f"""
            <div style="background:var(--bg-raised);border:1px solid var(--border);border-radius:8px;padding:14px 16px;font-family:'DM Sans',sans-serif;font-size:12.5px;line-height:1.7;color:var(--text-mid);">
                {r['interpretation']}
            </div>""", unsafe_allow_html=True)

    with col_r:
        if not row.empty:
            r   = row.iloc[0]
            vals = pd.to_numeric(
                fin_df[(fin_df["ticker"] == sel_ticker) & (fin_df["concept"] == sel_concept)]["value"],
                errors="coerce"
            ).dropna()

            from analysis.benfords_law import BenfordsAnalyzer, BENFORD
            analyzer = BenfordsAnalyzer()
            digits   = analyzer.extract_leading_digits(vals)
            obs_dist = analyzer.compute_distribution(digits)

            digits_x = list(range(1, 10))
            obs_y    = [obs_dist[d] * 100 for d in digits_x]
            exp_y    = [BENFORD[d] * 100   for d in digits_x]

            bar_color = "#ef4444" if r["flagged"] else "#22c55e"

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=digits_x, y=exp_y,
                name="Benford Expected",
                marker_color="rgba(255,255,255,.12)",
                marker_line=dict(color="rgba(255,255,255,.18)", width=1),
            ))
            fig.add_trace(go.Bar(
                x=digits_x, y=obs_y,
                name=f"{sel_ticker} Observed",
                marker_color=bar_color,
                marker_line=dict(color="rgba(0,0,0,.2)", width=0.5),
                opacity=0.85,
            ))
            fig.update_layout(
                title=dict(
                    text=f"{sel_ticker}  ·  {sel_concept}",
                    font=dict(family="DM Serif Display", size=15, color="#e8ecf4"),
                ),
                barmode="group",
                height=320,
                margin=dict(t=44, b=20, l=0, r=0),
                font=dict(family="DM Mono", size=11, color="#9ca3af"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#13161d",
                legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(title="Leading Digit", gridcolor="rgba(255,255,255,.06)", tickmode="linear"),
                yaxis=dict(title="Frequency %", gridcolor="rgba(255,255,255,.06)"),
            )
            st.plotly_chart(fig, use_container_width=True)

    divider()
    section_label("All Benford Results")

    display_cols = ["ticker","concept","n_values","mad","conformity_score","chi2_p","severity","flagged"]
    st.dataframe(
        benford_df[display_cols].style.apply(
            lambda row: ["background-color:rgba(239,68,68,.08)" if row["flagged"] else ""] * len(row),
            axis=1,
        ),
        use_container_width=True, height=320,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Isolation Forest":
    st.markdown("""
    <div class="page-head">
        <div class="page-head-kicker">ML · Multivariate Outlier Detection</div>
        <h1>Isolation <em>Forest</em></h1>
        <p>Multivariate anomaly detection across 32 financial features per company‑year.<br>
        Observations requiring fewer partitions to isolate are flagged as statistical outliers.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("Run an analysis from **Ingest & Analyze** first.")
        st.stop()

    import plotly.express as px
    import plotly.graph_objects as go

    if_df = st.session_state.if_df
    total = len(if_df)
    anom  = int(if_df["is_anomaly"].sum())

    c1, c2, c3 = st.columns(3)
    with c1: stat_tile(total, "Company-Years Scored")
    with c2: stat_tile(anom, "Anomalies Detected", "var(--red)")
    with c3: stat_tile(f"{anom/total*100:.1f}%" if total else "—", "Anomaly Rate", "var(--orange)")

    divider()

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        section_label("Anomaly Score Distribution")
        fig = px.scatter(
            if_df.sort_values("if_score_norm", ascending=False),
            x="fiscal_year", y="if_score_norm",
            color="is_anomaly",
            color_discrete_map={True: "#ef4444", False: "rgba(255,255,255,.2)"},
            hover_data=["ticker", "top_features"],
            labels={"if_score_norm": "Anomaly Score (0–100)", "fiscal_year": "Fiscal Year"},
        )
        fig.update_layout(
            height=340,
            margin=dict(t=20, b=20, l=0, r=0),
            font=dict(family="DM Mono", size=11, color="#9ca3af"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#13161d",
            legend=dict(font=dict(family="DM Mono", size=10), bgcolor="rgba(0,0,0,0)", title_text="Anomaly"),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,.06)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,.06)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_label("Top Anomalies")
        top_anom = if_df[if_df["is_anomaly"]].sort_values("if_score_norm", ascending=False).head(10)
        for _, r in top_anom.iterrows():
            sev = "critical" if r["if_score_norm"] >= 75 else "high" if r["if_score_norm"] >= 50 else "medium"
            st.markdown(f"""
            <div class="flag flag-{sev}">
                <strong>{r['ticker']} · FY{r['fiscal_year']}</strong>
                Score: {r['if_score_norm']:.0f} / 100<br>
                <span style="font-size:11px;color:var(--text-muted);">{r['top_features']}</span>
            </div>""", unsafe_allow_html=True)

    divider()
    section_label("Full Scores Table")
    st.dataframe(
        if_df[["ticker","fiscal_year","if_score_norm","is_anomaly","top_features","explanation"]]
        .sort_values("if_score_norm", ascending=False),
        use_container_width=True, height=360,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB: YoY Variance
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "YoY Variance":
    st.markdown("""
    <div class="page-head">
        <div class="page-head-kicker">Statistical · Threshold Analysis</div>
        <h1>YoY <em>Variance</em></h1>
        <p>Year-over-year change detection with Z-score normalisation, hard thresholds,<br>
        sign-reversal detection, and accrual ratio analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("Run an analysis from **Ingest & Analyze** first.")
        st.stop()

    import plotly.graph_objects as go

    var_df  = st.session_state.variance_df
    fin_df  = st.session_state.fin_df
    tickers = st.session_state.tickers_used

    sev_counts = var_df["severity"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_tile(sev_counts.get("critical", 0), "Critical", "var(--red)")
    with c2: stat_tile(sev_counts.get("high",     0), "High",     "var(--orange)")
    with c3: stat_tile(sev_counts.get("medium",   0), "Medium",   "var(--yellow)")
    with c4: stat_tile(sev_counts.get("low",      0), "Low",      "var(--green)")

    divider()

    col_l, col_r = st.columns([2, 3], gap="large")
    with col_l:
        section_label("Filters")
        sel_ticker  = st.selectbox("Company", ["All"] + tickers, label_visibility="collapsed")
        sev_filter  = st.multiselect(
            "Severity", ["critical","high","medium","low"],
            default=["critical","high"], label_visibility="collapsed",
        )
        flag_filter = st.multiselect(
            "Flag Type",
            ["outlier_zscore","hard_threshold","sign_reversal","accrual"],
            default=["outlier_zscore","hard_threshold","sign_reversal","accrual"],
            label_visibility="collapsed",
        )

    filtered = var_df.copy()
    if sel_ticker != "All":
        filtered = filtered[filtered["ticker"] == sel_ticker]
    filtered = filtered[
        filtered["severity"].isin(sev_filter) &
        filtered["flag_type"].isin(flag_filter)
    ]

    with col_r:
        section_label(f"{len(filtered)} flags")
        for _, row in filtered.head(15).iterrows():
            st.markdown(f"""
            <div class="flag flag-{row['severity']}">
                <strong>{row['ticker']} · {row['concept']} · FY{row['fiscal_year']} · {row['flag_type'].replace('_',' ').upper()}</strong>
                {row['explanation']}
            </div>""", unsafe_allow_html=True)

    divider()

    if not filtered.empty:
        section_label("YoY Change Heatmap")
        pivot = filtered.pivot_table(
            index="concept", columns="fiscal_year",
            values="yoy_pct", aggfunc="mean"
        ).fillna(0)
        if not pivot.empty:
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=list(pivot.index),
                colorscale=[[0,"#ef4444"],[0.5,"#1a1e28"],[1,"#22c55e"]],
                zmid=0,
                text=pivot.values.round(1),
                texttemplate="%{text}%",
                colorbar=dict(
                    title="YoY %",
                    tickfont=dict(family="DM Mono", size=10, color="#9ca3af"),
                    titlefont=dict(family="DM Mono", size=10, color="#9ca3af"),
                ),
            ))
            fig.update_layout(
                height=max(300, len(pivot) * 30),
                margin=dict(t=20, b=20, l=0, r=0),
                font=dict(family="DM Mono", size=10, color="#9ca3af"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#13161d",
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB: Scorecard Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Scorecard Dashboard":
    st.markdown("""
    <div class="page-head">
        <div class="page-head-kicker">Composite · Risk Summary</div>
        <h1>Scorecard <em>Dashboard</em></h1>
        <p>Company health scorecard aggregating Benford's Law, Isolation Forest,<br>
        and YoY variance signals into a composite anomaly risk score.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.info("Run an analysis from **Ingest & Analyze** first.")
        st.stop()

    import plotly.graph_objects as go
    from reporting.report_generator import ReportGenerator

    sc_df    = st.session_state.sc_df
    reporter = ReportGenerator()

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_tile(sc_df["ticker"].nunique(), "Companies")
    with c2: stat_tile(int((sc_df["risk_level"].isin(["critical","high"])).sum()), "Critical / High", "var(--red)")
    with c3: stat_tile(f"{sc_df['composite_score'].mean():.1f}", "Avg Composite Score", "var(--amber)")
    with c4: stat_tile(int((sc_df["risk_level"] == "none").sum()), "Clean Companies", "var(--green)")

    divider()

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        section_label("Composite Score by Company")
        agg = sc_df.groupby("ticker").agg(
            composite_score=("composite_score", "max"),
            risk_level=("risk_level", lambda x: x.value_counts().index[0]),
        ).reset_index().sort_values("composite_score", ascending=True)

        color_map = {"critical":"#ef4444","high":"#f97316","medium":"#eab308",
                     "low":"#22c55e","none":"rgba(255,255,255,.2)"}
        agg["color"] = agg["risk_level"].map(color_map)

        fig = go.Figure(go.Bar(
            x=agg["composite_score"], y=agg["ticker"],
            orientation="h",
            marker_color=agg["color"],
            marker_line=dict(color="rgba(0,0,0,.15)", width=0.5),
        ))
        fig.add_vline(
            x=50, line_dash="dash", line_color="#ef4444", opacity=.6,
            annotation_text="High risk threshold",
            annotation_font=dict(family="DM Mono", size=10, color="#ef4444"),
        )
        fig.update_layout(
            height=max(300, len(agg) * 34),
            margin=dict(t=20, b=20, l=0, r=20),
            font=dict(family="DM Mono", size=11, color="#9ca3af"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#13161d",
            xaxis=dict(title="Composite Anomaly Score (0–100)", range=[0,100], gridcolor="rgba(255,255,255,.06)"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_label("Risk Distribution")
        rc = sc_df.groupby("risk_level")["ticker"].nunique().reset_index()
        rc.columns = ["risk_level", "count"]
        order = ["critical","high","medium","low","none"]
        rc["risk_level"] = pd.Categorical(rc["risk_level"], categories=order, ordered=True)
        rc = rc.sort_values("risk_level")

        fig2 = go.Figure(go.Pie(
            labels=rc["risk_level"].str.title(),
            values=rc["count"],
            hole=0.58,
            marker_colors=["#ef4444","#f97316","#eab308","#22c55e","rgba(255,255,255,.15)"],
            marker_line=dict(color="#0d0f14", width=2),
            textfont=dict(family="DM Mono", size=11),
        ))
        fig2.update_layout(
            height=260,
            margin=dict(t=20, b=0, l=0, r=0),
            font=dict(family="DM Mono"),
            showlegend=True,
            legend=dict(font=dict(family="DM Mono", size=10, color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    divider()
    section_label("Full Scorecard Table")
    display = sc_df[[
        "ticker","fiscal_year","composite_score","risk_level",
        "benford_score","if_score","variance_score","accrual_score",
        "benford_flags","variance_flags","if_anomalies",
    ]].sort_values("composite_score", ascending=False)
    st.dataframe(display, use_container_width=True, height=400)

    divider()
    section_label("Power BI Export")
    col_e1, col_e2 = st.columns([2, 3])
    with col_e1:
        if st.button("Export Power BI Files →"):
            outputs = reporter.export_powerbi(
                st.session_state.scorecards,
                benford_df  = st.session_state.benford_df,
                variance_df = st.session_state.variance_df,
                if_df       = st.session_state.if_df,
            )
            st.success(f"✓  Exported {len(outputs)} files")
            for name, path in outputs.items():
                st.caption(f"{name}: `{path}`")
    with col_e2:
        st.markdown("""
        <div class="info-box">
            <strong>Power BI Setup</strong>
            1. Export → import <code>powerbi_scorecards.csv</code> as main table<br>
            2. Join <code>powerbi_variance_flags.csv</code> on ticker + fiscal_year<br>
            3. Join <code>powerbi_benford.csv</code> on ticker<br>
            4. Use <code>composite_score</code> for conditional formatting<br>
            5. See data dictionary for all DAX measure recommendations
        </div>
        """, unsafe_allow_html=True)