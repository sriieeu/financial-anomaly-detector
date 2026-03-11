"""
report_generator.py — Company health scorecard and Power BI export.

Aggregates signals from:
  - Benford's Law (digit distribution)
  - Isolation Forest (multivariate ML)
  - Year-over-year variance analysis
  - Accrual ratio

into a unified risk scorecard and Power BI–ready CSV export.

Usage:
    gen = ReportGenerator()
    scorecard = gen.build_scorecard(ticker, benford_df, if_df, variance_df)
    gen.export_powerbi(all_scorecards, "data/processed/powerbi_export.csv")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


@dataclass
class CompanyScorecard:
    """Unified risk scorecard for a single company."""
    ticker: str
    company_name: str
    fiscal_year: int

    # Component scores (0–100, higher = more suspicious)
    benford_score:  float = 0.0
    if_score:       float = 0.0
    variance_score: float = 0.0
    accrual_score:  float = 0.0

    # Composite
    composite_score: float = 0.0
    risk_level: str = "none"     # none / low / medium / high / critical

    # Flags
    benford_flags:  int = 0
    variance_flags: int = 0
    if_anomalies:   int = 0

    # Details
    top_concerns: list = field(default_factory=list)
    notes: str = ""


class ReportGenerator:
    """
    Aggregates multi-signal anomaly scores into company health scorecards.

    Weighting scheme (tunable):
        Benford's Law:      25%
        Isolation Forest:   35%
        YoY Variance:       30%
        Accrual Ratio:      10%
    """

    WEIGHTS = {
        "benford":  0.25,
        "if":       0.35,
        "variance": 0.30,
        "accrual":  0.10,
    }

    def build_scorecard(
        self,
        ticker: str,
        company_name: str,
        fiscal_year: int,
        benford_df:  Optional[pd.DataFrame] = None,
        if_df:       Optional[pd.DataFrame] = None,
        variance_df: Optional[pd.DataFrame] = None,
    ) -> CompanyScorecard:
        """
        Build a unified scorecard for one (company, fiscal_year).

        Args:
            ticker:       Stock ticker
            company_name: Full company name
            fiscal_year:  Year to score
            benford_df:   Output from BenfordsAnalyzer.analyze_dataframe()
            if_df:        Output from AnomalyDetector.score_all()
            variance_df:  Output from VarianceAnalyzer.flag_all()
        """
        sc = CompanyScorecard(
            ticker=ticker,
            company_name=company_name,
            fiscal_year=fiscal_year,
        )

        concerns = []

        # ── Benford component ──────────────────────────────────────────────
        if benford_df is not None and not benford_df.empty:
            t_bend = benford_df[benford_df["ticker"] == ticker]
            if not t_bend.empty:
                flagged   = t_bend[t_bend["flagged"] == True]
                sc.benford_flags = len(flagged)
                max_mad = t_bend["mad"].max()

                # Score: scale MAD to 0–100 (MAD=0.05 → 100)
                sc.benford_score = min(100.0, float(max_mad) / 0.05 * 100)

                for _, row in flagged.iterrows():
                    concerns.append(
                        f"Benford [{row['severity'].upper()}] {row['concept']}: "
                        f"MAD={row['mad']:.4f}"
                    )

        # ── Isolation Forest component ─────────────────────────────────────
        if if_df is not None and not if_df.empty:
            t_if = if_df[
                (if_df["ticker"] == ticker) &
                (if_df["fiscal_year"] == fiscal_year)
            ]
            if not t_if.empty:
                sc.if_score      = float(t_if["if_score_norm"].max())
                sc.if_anomalies  = int(t_if["is_anomaly"].sum())
                if sc.if_anomalies > 0:
                    concerns.append(
                        f"Isolation Forest: FY{fiscal_year} flagged as statistical outlier "
                        f"(score={sc.if_score:.0f}/100)"
                    )

        # ── Variance component ─────────────────────────────────────────────
        if variance_df is not None and not variance_df.empty:
            t_var = variance_df[
                (variance_df["ticker"] == ticker) &
                (variance_df["fiscal_year"] == fiscal_year)
            ]
            if not t_var.empty:
                sc.variance_flags = len(t_var)

                # Score: weighted by severity
                sev_scores = {"critical": 100, "high": 70, "medium": 40, "low": 15}
                sev_vals   = [sev_scores.get(s, 0) for s in t_var["severity"]]
                sc.variance_score = min(100.0, float(np.mean(sev_vals))) if sev_vals else 0.0

                for _, row in t_var.iterrows():
                    concerns.append(
                        f"Variance [{row['severity'].upper()}] {row['concept']}: "
                        f"{row['yoy_pct']:+.1f}% YoY (FY{fiscal_year})"
                    )

        # ── Accrual sub-score (extracted from variance flags) ──────────────
        if variance_df is not None and not variance_df.empty:
            accrual_flags = variance_df[
                (variance_df["ticker"] == ticker) &
                (variance_df["flag_type"] == "accrual") &
                (variance_df["fiscal_year"] == fiscal_year)
            ]
            if not accrual_flags.empty:
                sc.accrual_score = 80.0 if "critical" in accrual_flags["severity"].values else 50.0
                concerns.append(
                    f"Accrual ratio flag: earnings quality concern in FY{fiscal_year}"
                )

        # ── Composite score ────────────────────────────────────────────────
        sc.composite_score = round(
            sc.benford_score  * self.WEIGHTS["benford"] +
            sc.if_score       * self.WEIGHTS["if"]      +
            sc.variance_score * self.WEIGHTS["variance"] +
            sc.accrual_score  * self.WEIGHTS["accrual"],
            1,
        )

        sc.risk_level = (
            "critical" if sc.composite_score >= 75 else
            "high"     if sc.composite_score >= 50 else
            "medium"   if sc.composite_score >= 25 else
            "low"      if sc.composite_score >= 10 else
            "none"
        )

        sc.top_concerns = concerns[:5]
        return sc

    def build_all_scorecards(
        self,
        tickers: list,
        company_names: dict,   # {ticker: company_name}
        fiscal_years: list,
        benford_df:  Optional[pd.DataFrame] = None,
        if_df:       Optional[pd.DataFrame] = None,
        variance_df: Optional[pd.DataFrame] = None,
    ) -> list:
        """Build scorecards for all (ticker, year) combinations."""
        scorecards = []
        for ticker in tickers:
            name = company_names.get(ticker, ticker)
            years = fiscal_years if fiscal_years else [2023]
            for year in years:
                sc = self.build_scorecard(
                    ticker, name, year,
                    benford_df, if_df, variance_df
                )
                scorecards.append(sc)
        return sorted(scorecards, key=lambda x: x.composite_score, reverse=True)

    def scorecards_to_dataframe(self, scorecards: list) -> pd.DataFrame:
        """Convert list of CompanyScorecard to flat DataFrame."""
        return pd.DataFrame([{
            "ticker":          s.ticker,
            "company_name":    s.company_name,
            "fiscal_year":     s.fiscal_year,
            "composite_score": s.composite_score,
            "risk_level":      s.risk_level,
            "benford_score":   round(s.benford_score, 1),
            "if_score":        round(s.if_score, 1),
            "variance_score":  round(s.variance_score, 1),
            "accrual_score":   round(s.accrual_score, 1),
            "benford_flags":   s.benford_flags,
            "variance_flags":  s.variance_flags,
            "if_anomalies":    s.if_anomalies,
            "top_concerns":    " | ".join(s.top_concerns),
        } for s in scorecards])

    def export_powerbi(
        self,
        scorecards: list,
        output_path: str = "data/processed/powerbi_export.csv",
        benford_df:  Optional[pd.DataFrame] = None,
        variance_df: Optional[pd.DataFrame] = None,
        if_df:       Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Export Power BI–ready files.

        Produces:
          - powerbi_scorecards.csv  — company health scorecard (main table)
          - powerbi_flags.csv       — all individual flags (detail table)
          - powerbi_benford.csv     — Benford digit distributions
          - powerbi_readme.txt      — data dictionary

        Returns dict of {table_name: filepath}
        """
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}

        # 1. Scorecards
        sc_df  = self.scorecards_to_dataframe(scorecards)
        sc_path = out_dir / "powerbi_scorecards.csv"
        sc_df.to_csv(sc_path, index=False)
        outputs["scorecards"] = str(sc_path)
        logger.info(f"Exported {len(sc_df)} scorecards → {sc_path}")

        # 2. Variance flags
        if variance_df is not None and not variance_df.empty:
            vf_path = out_dir / "powerbi_variance_flags.csv"
            variance_df.to_csv(vf_path, index=False)
            outputs["variance_flags"] = str(vf_path)

        # 3. Benford results
        if benford_df is not None and not benford_df.empty:
            bf_path = out_dir / "powerbi_benford.csv"
            benford_df.to_csv(bf_path, index=False)
            outputs["benford"] = str(bf_path)

        # 4. Isolation Forest scores
        if if_df is not None and not if_df.empty:
            if_path = out_dir / "powerbi_isolation_forest.csv"
            if_df.to_csv(if_path, index=False)
            outputs["isolation_forest"] = str(if_path)

        # 5. Data dictionary
        readme = out_dir / "powerbi_data_dictionary.txt"
        readme.write_text(POWERBI_DATA_DICT)
        outputs["readme"] = str(readme)

        logger.info(f"Power BI export complete: {len(outputs)} files")
        return outputs


POWERBI_DATA_DICT = """
POWER BI DATA DICTIONARY — Financial Statement Anomaly Detector
================================================================

TABLE: powerbi_scorecards.csv
  ticker            — Stock ticker symbol (join key)
  company_name      — Full legal company name
  fiscal_year       — Fiscal year (integer)
  composite_score   — Overall anomaly score 0–100 (100 = most suspicious)
  risk_level        — Categorical: none / low / medium / high / critical
  benford_score     — Benford's Law component score 0–100
  if_score          — Isolation Forest component score 0–100
  variance_score    — YoY variance component score 0–100
  accrual_score     — Accrual ratio component score 0–100
  benford_flags     — Count of Benford's Law flag events
  variance_flags    — Count of YoY variance flag events
  if_anomalies      — Count of Isolation Forest anomaly flags
  top_concerns      — Pipe-delimited list of top risk concerns

TABLE: powerbi_variance_flags.csv
  ticker            — Stock ticker
  concept           — Financial line item (e.g. "Revenues", "NetIncomeLoss")
  fiscal_year       — Year of the observation
  yoy_pct           — Year-over-year % change
  zscore            — Z-score vs peer/historical distribution
  flag_type         — outlier_zscore | hard_threshold | sign_reversal | accrual
  severity          — low / medium / high / critical
  explanation       — Plain-English description of the flag

TABLE: powerbi_benford.csv
  ticker            — Stock ticker
  concept           — Financial concept analyzed
  n_values          — Sample size
  mad               — Mean Absolute Deviation from Benford's distribution
  conformity_score  — 0–100 (100 = perfect Benford conformity)
  chi2_stat         — Chi-square test statistic
  chi2_p            — Chi-square p-value (< 0.05 = significant deviation)
  severity          — none / low / medium / high / critical
  flagged           — Boolean: True = warrants review
  interpretation    — Plain-English explanation

TABLE: powerbi_isolation_forest.csv
  ticker            — Stock ticker
  fiscal_year       — Fiscal year
  if_score_raw      — Raw Isolation Forest score (-1 to 0; more negative = anomalous)
  if_score_norm     — Normalised score 0–100 (100 = most anomalous)
  is_anomaly        — Boolean: True = flagged as outlier
  top_features      — Top contributing financial features
  explanation       — Plain-English explanation

COMPOSITE SCORE METHODOLOGY
  composite = 0.25 × benford + 0.35 × isolation_forest + 0.30 × variance + 0.10 × accrual
  Weights reflect empirical effectiveness in forensic accounting literature.

RECOMMENDED POWER BI MEASURES
  High Risk Count = COUNTROWS(FILTER(scorecards, [risk_level] = "high" || [risk_level] = "critical"))
  Avg Composite   = AVERAGE(scorecards[composite_score])
  Flag Rate       = DIVIDE([High Risk Count], COUNTROWS(scorecards))
"""
