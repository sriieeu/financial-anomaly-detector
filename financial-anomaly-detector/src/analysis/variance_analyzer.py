"""
variance_analyzer.py — Year-over-year variance analysis with auto-flagging.

Detects unusual changes in financial line items using:
1. YoY % change with Z-score normalisation across peer companies
2. Absolute magnitude thresholds per concept
3. Reversal detection (positive→negative or vice versa)
4. Accrual ratio analysis (earnings quality)

Usage:
    analyzer = VarianceAnalyzer()
    results  = analyzer.analyze(df, ticker="AAPL")
    flags    = analyzer.flag_all(df)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Z-score threshold for flagging outlier YoY changes
ZSCORE_THRESHOLD = 2.5

# Absolute YoY % change thresholds that always flag regardless of Z-score
HARD_THRESHOLDS = {
    "Revenues":                                   0.50,   # >50% revenue jump
    "RevenueFromContractWithCustomerExcludingAssessedTax": 0.50,
    "NetIncomeLoss":                              1.00,   # >100% net income swing
    "Assets":                                     0.40,
    "Liabilities":                                0.40,
    "LongTermDebt":                               0.75,
    "AccountsReceivableNetCurrent":               0.60,
    "InventoryNet":                               0.60,
    "CashAndCashEquivalentsAtCarryingValue":       0.80,
    "GoodwillAndIntangibleAssetsNetOfAmortization": 0.50,
}

# Concepts where sign reversal (positive↔negative) is always flagged
SIGN_REVERSAL_CONCEPTS = [
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "RetainedEarningsAccumulatedDeficit",
    "NetCashProvidedByUsedInOperatingActivities",
]


@dataclass
class VarianceFlag:
    ticker: str
    concept: str
    fiscal_year: int
    prior_year: int
    current_value: float
    prior_value: float
    yoy_change_pct: float
    zscore: float
    flag_type: str          # "outlier_zscore" | "hard_threshold" | "sign_reversal" | "accrual"
    severity: str           # low / medium / high / critical
    explanation: str


@dataclass
class VarianceResult:
    ticker: str
    flags: list             # List[VarianceFlag]
    summary_df: pd.DataFrame
    anomaly_score: float    # 0–100
    risk_level: str


class VarianceAnalyzer:
    """
    Year-over-year financial variance analysis with statistical auto-flagging.

    Compares each company's changes against:
    1. Its own historical volatility (individual Z-score)
    2. Cross-sectional peer distribution (peer Z-score)
    3. Absolute hard thresholds per financial concept
    4. Sign reversals in key profitability metrics
    """

    def analyze(
        self,
        df: pd.DataFrame,
        ticker: str,
        peer_df: Optional[pd.DataFrame] = None,
    ) -> VarianceResult:
        """
        Analyse YoY variance for a single company.

        Args:
            df:       DataFrame with cols [ticker, concept, end_date, value, fiscal_year]
            ticker:   Target ticker
            peer_df:  Full multi-company dataset for peer comparison (optional)
        """
        company_df = df[df["ticker"] == ticker].copy()
        if company_df.empty:
            return VarianceResult(ticker, [], pd.DataFrame(), 0.0, "unknown")

        flags   = []
        summary = []

        for concept, grp in company_df.groupby("concept"):
            grp = grp.sort_values("end_date").drop_duplicates("fiscal_year")
            vals = grp.set_index("fiscal_year")["value"]

            if len(vals) < 2:
                continue

            # Compute YoY changes
            yoy = vals.pct_change().dropna()

            # Peer Z-scores (if peer data provided)
            peer_changes = None
            if peer_df is not None:
                peer_grp = peer_df[peer_df["concept"] == concept]
                peer_yoy = (
                    peer_grp.sort_values(["ticker", "end_date"])
                    .groupby("ticker")["value"]
                    .pct_change()
                    .dropna()
                )
                if len(peer_yoy) >= 5:
                    peer_changes = peer_yoy

            for year, pct_change in yoy.items():
                if not np.isfinite(pct_change):
                    continue

                prior_year = year - 1
                if prior_year not in vals.index:
                    continue

                current_val = vals[year]
                prior_val   = vals[prior_year]

                # Z-score vs peers (or vs own history)
                ref_series = peer_changes if peer_changes is not None else yoy
                if len(ref_series) >= 3:
                    mu  = ref_series.mean()
                    sig = ref_series.std()
                    z   = (pct_change - mu) / sig if sig > 0 else 0.0
                else:
                    z = 0.0

                row_flags = []

                # ── Test 1: Z-score outlier ──
                if abs(z) >= ZSCORE_THRESHOLD:
                    sev = "critical" if abs(z) >= 4.0 else "high" if abs(z) >= 3.0 else "medium"
                    row_flags.append(VarianceFlag(
                        ticker=ticker, concept=str(concept),
                        fiscal_year=int(year), prior_year=int(prior_year),
                        current_value=float(current_val), prior_value=float(prior_val),
                        yoy_change_pct=float(pct_change), zscore=float(z),
                        flag_type="outlier_zscore", severity=sev,
                        explanation=(
                            f"{concept} changed {pct_change*100:+.1f}% in FY{year} "
                            f"(Z={z:+.2f} vs {'peers' if peer_changes is not None else 'own history'}). "
                            f"This is a {abs(z):.1f}σ event — statistically unusual."
                        ),
                    ))

                # ── Test 2: Hard threshold ──
                threshold = HARD_THRESHOLDS.get(str(concept))
                if threshold and abs(pct_change) >= threshold:
                    sev = "high" if abs(pct_change) < threshold * 2 else "critical"
                    row_flags.append(VarianceFlag(
                        ticker=ticker, concept=str(concept),
                        fiscal_year=int(year), prior_year=int(prior_year),
                        current_value=float(current_val), prior_value=float(prior_val),
                        yoy_change_pct=float(pct_change), zscore=float(z),
                        flag_type="hard_threshold", severity=sev,
                        explanation=(
                            f"{concept} changed {pct_change*100:+.1f}% in FY{year}, "
                            f"exceeding the {threshold*100:.0f}% alert threshold."
                        ),
                    ))

                # ── Test 3: Sign reversal ──
                if str(concept) in SIGN_REVERSAL_CONCEPTS:
                    if (prior_val > 0 and current_val < 0) or (prior_val < 0 and current_val > 0):
                        row_flags.append(VarianceFlag(
                            ticker=ticker, concept=str(concept),
                            fiscal_year=int(year), prior_year=int(prior_year),
                            current_value=float(current_val), prior_value=float(prior_val),
                            yoy_change_pct=float(pct_change), zscore=float(z),
                            flag_type="sign_reversal", severity="high",
                            explanation=(
                                f"{concept} reversed from "
                                f"{'profit' if prior_val > 0 else 'loss'} to "
                                f"{'loss' if current_val < 0 else 'profit'} in FY{year}."
                            ),
                        ))

                flags.extend(row_flags)

                summary.append({
                    "ticker":        ticker,
                    "concept":       concept,
                    "fiscal_year":   year,
                    "prior_year":    prior_year,
                    "current_value": current_val,
                    "prior_value":   prior_val,
                    "yoy_pct":       round(pct_change * 100, 2),
                    "zscore":        round(z, 3),
                    "n_flags":       len(row_flags),
                    "flagged":       len(row_flags) > 0,
                })

        # ── Accrual ratio analysis ──
        accrual_flags = self._accrual_analysis(company_df, ticker)
        flags.extend(accrual_flags)

        # ── Anomaly score ──
        sev_weights = {"low": 1, "medium": 3, "high": 6, "critical": 10}
        raw_score   = sum(sev_weights.get(f.severity, 0) for f in flags)
        norm_score  = min(100.0, raw_score * 5)

        risk_level = (
            "critical" if norm_score >= 80 else
            "high"     if norm_score >= 50 else
            "medium"   if norm_score >= 25 else
            "low"      if norm_score > 0  else "none"
        )

        return VarianceResult(
            ticker=ticker,
            flags=flags,
            summary_df=pd.DataFrame(summary),
            anomaly_score=round(norm_score, 1),
            risk_level=risk_level,
        )

    def flag_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run variance analysis on all tickers in df.
        Returns a flat DataFrame of all flags.
        """
        all_flags = []
        tickers   = df["ticker"].unique()

        for ticker in tickers:
            result = self.analyze(df, ticker=str(ticker), peer_df=df)
            for f in result.flags:
                all_flags.append({
                    "ticker":       f.ticker,
                    "concept":      f.concept,
                    "fiscal_year":  f.fiscal_year,
                    "yoy_pct":      round(f.yoy_change_pct * 100, 2),
                    "zscore":       round(f.zscore, 3),
                    "flag_type":    f.flag_type,
                    "severity":     f.severity,
                    "explanation":  f.explanation,
                })

        return pd.DataFrame(all_flags).sort_values(
            ["severity", "zscore"],
            ascending=[True, False],
            key=lambda col: col.map({"critical": 0, "high": 1, "medium": 2, "low": 3})
            if col.name == "severity" else col,
        )

    def _accrual_ratio(
        self,
        net_income: float,
        op_cashflow: float,
        avg_assets: float,
    ) -> float:
        """
        Accrual ratio = (Net Income − Operating Cash Flow) / Avg Total Assets

        High accrual ratio (> 0.10) indicates earnings quality concerns —
        income is driven by non-cash accruals rather than actual cash generation.
        Widely used in forensic accounting (Sloan 1996 accrual anomaly).
        """
        if avg_assets == 0:
            return 0.0
        return (net_income - op_cashflow) / avg_assets

    def _accrual_analysis(
        self, df: pd.DataFrame, ticker: str
    ) -> list:
        """Compute accrual ratio and flag high-accrual years."""
        flags = []

        try:
            ni  = df[df["concept"] == "NetIncomeLoss"].set_index("fiscal_year")["value"]
            ocf = df[df["concept"] == "NetCashProvidedByUsedInOperatingActivities"].set_index("fiscal_year")["value"]
            ast = df[df["concept"] == "Assets"].set_index("fiscal_year")["value"]

            common_years = ni.index.intersection(ocf.index).intersection(ast.index)
            for year in common_years:
                avg_assets = ast[year]
                if avg_assets <= 0:
                    continue
                ratio = self._accrual_ratio(ni[year], ocf[year], avg_assets)
                if abs(ratio) > 0.10:
                    sev = "critical" if abs(ratio) > 0.20 else "high"
                    flags.append(VarianceFlag(
                        ticker=ticker, concept="AccrualRatio",
                        fiscal_year=int(year), prior_year=int(year) - 1,
                        current_value=float(ratio), prior_value=0.0,
                        yoy_change_pct=0.0, zscore=0.0,
                        flag_type="accrual", severity=sev,
                        explanation=(
                            f"Accrual ratio = {ratio:.3f} in FY{year} "
                            f"(threshold: ±0.10). High accruals indicate "
                            f"earnings driven by non-cash items — an earnings quality red flag."
                        ),
                    ))
        except Exception as e:
            logger.debug(f"Accrual analysis skipped for {ticker}: {e}")

        return flags
