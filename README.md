Automated Financial Statement Anomaly Detector


Public companies are required to periodically disclose their financial performance through regulatory filings such as annual reports (10-K) and quarterly reports (10-Q). These documents contain detailed financial statements including balance sheets, income statements, and cash flow statements. Because these reports influence investors, regulators, and financial markets, it is essential that the information they contain is accurate and free from manipulation.
However, detecting irregularities in financial statements manually is difficult. Financial reports can span hundreds of pages and contain thousands of numerical values. Forensic accountants and regulators therefore use statistical and machine learning techniques to identify patterns that may indicate possible accounting manipulation or reporting inconsistencies.
The Automated Financial Statement Anomaly Detector is a system designed to automatically analyze financial data from regulatory filings and detect unusual patterns that may signal potential irregularities. The system collects filings from the SEC EDGAR database, extracts financial tables, and applies statistical and machine learning methods such as Benford’s Law and Isolation Forest to detect anomalies in reported numbers.

 Objectives

The primary goals of the Automated Financial Statement Anomaly Detector are:
Automatically collect corporate financial filings from the SEC EDGAR database.
Extract financial tables from 10-K and 10-Q reports.
Analyze financial figures using statistical fraud detection techniques.
Identify unusual numeric patterns that deviate from expected distributions.
Detect anomalies using machine learning algorithms.
Provide analysts with potential red flags for further investigation.

## Architecture

```
financial-anomaly-detector/
├── src/
│   ├── ingestion/
│   │   ├── edgar_client.py        # SEC EDGAR XBRL API client (rate-limited, cached)
│   │   └── filing_downloader.py   # Batch multi-company ingestion pipeline
│   ├── parsing/
│   │   └── pdf_parser.py          # pdfplumber table extraction from 10-K PDFs
│   ├── analysis/
│   │   ├── benfords_law.py        # Benford's Law + MAD + chi-square analysis
│   │   └── variance_analyzer.py   # YoY variance, Z-score, accrual ratio
│   ├── models/
│   │   └── isolation_forest.py    # Multivariate anomaly detection (sklearn)
│   ├── reporting/
│   │   └── report_generator.py    # Scorecard + Power BI export
│   └── ui/
│       └── app.py                 # Streamlit five-tab dashboard
├── tests/
│   └── test_pipeline.py           # 22 pytest tests
├── data/
│   ├── raw/                       # Downloaded SEC filings
│   ├── processed/                 # Parsed datasets + Power BI exports
│   └── cache/                     # Per-company EDGAR cache (Parquet)
├── requirements.txt
└── packages.txt                   # Streamlit Cloud system deps
```

## Setup

```bash
pip install -r requirements.txt
streamlit run src/ui/app.py
pytest tests/ -v
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| EDGAR Ingestion | `data.sec.gov` XBRL API — 32 financial concepts, rate-limited to 6.7 req/s |
| Benford's Law | MAD + chi-square vs expected digit distribution; Nigrini (2012) thresholds |
| Isolation Forest | 200 trees, RobustScaler, 30+ derived features including financial ratios |
| YoY Variance | Z-score vs peer universe + hard thresholds + sign-reversal detection |
| Accrual Ratio | Sloan (1996) earnings quality metric |
| Composite Score | Weighted: Benford 25% · IF 35% · Variance 30% · Accrual 10% |
| Power BI Export | 4 linked CSV tables with full data dictionary |

## Why Benford's Law Works

Benford's Law states that in naturally occurring numerical data, the leading digit 1 appears ~30% of the time, 2 ~17.6%, and so on logarithmically. Fabricated numbers cluster around "round" figures (50,000, 100,000) producing detectable deviations. The IRS uses this in every major tax fraud investigation. Applying it at scale across 500+ SEC filings with ML cross-validation is genuinely sophisticated forensic analytics.

## Tech Stack

`Python` · `SEC EDGAR API` · `pdfplumber` · `pandas` · `scikit-learn` · `scipy` · `Streamlit` · `Plotly` · `Power BI`
