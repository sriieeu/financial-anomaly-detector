"""
benfords_law.py — Benford's Law digit distribution analysis.

Benford's Law: In naturally occurring financial data, the leading digit d
appears with probability P(d) = log10(1 + 1/d).

Expected:  1→30.1%  2→17.6%  3→12.5%  4→9.7%  5→7.9%
           6→6.7%   7→5.8%   8→5.1%   9→4.6%

Used by: IRS, SEC enforcement, Big 4 forensic accounting, PCAOB auditors.
Reference: Nigrini (2012) "Benford's Law: Applications for Forensic Accounting"

Usage:
    analyzer = BenfordsAnalyzer()
    result   = analyzer.analyze(values, ticker="AAPL", concept="Revenues")
    df       = analyzer.analyze_dataframe(full_df)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Benford's expected probabilities for digits 1–9
BENFORD = {d: np.log10(1 + 1 / d) for d in range(1, 10)}
BENFORD_ARR = np.array([BENFORD[d] for d in range(1, 10)])

# Nigrini (2012) MAD thresholds for first-digit test
MAD_THRESHOLDS = {
    "none":     0.006,   # Close conformity
    "low":      0.012,   # Acceptable
    "medium":   0.015,   # Marginally acceptable
    # > 0.015 = non-conformity
}


@dataclass
class BenfordsResult:
    ticker: str
    concept: str
    n_values: int
    observed: dict          # {1: 0.28, ..., 9: 0.04}
    expected: dict          # Benford expected
    mad: float              # Mean Absolute Deviation
    chi2_stat: float
    chi2_p: float
    conformity_score: float # 0–100 (100 = perfect Benford)
    severity: str           # none / low / medium / high / critical
    flagged: bool
    top_deviating: list     # digits with largest deviation
    interpretation: str


class BenfordsAnalyzer:
    """
    Applies Benford's Law analysis to financial datasets.

    Two test statistics:
    1. MAD (Mean Absolute Deviation) — Nigrini's preferred metric
    2. Chi-square goodness-of-fit — classical statistical test

    Both are computed; MAD drives the flag/severity logic because
    chi-square is sensitive to sample size and can over-flag large datasets.
    """

    def extract_leading_digits(self, values: pd.Series) -> pd.Series:
        """
        Extract first significant digit from each value.
        - Ignores zeros, negatives, NaN
        - Handles decimals: 0.0034 → 3
        """
        digits = []
        for val in values:
            try:
                v = abs(float(val))
            except (TypeError, ValueError):
                continue
            if v <= 0 or np.isnan(v) or np.isinf(v):
                continue
            # Normalise: shift until 1 ≤ v < 10
            while v < 1:
                v *= 10
            while v >= 10:
                v /= 10
            d = int(v)
            if 1 <= d <= 9:
                digits.append(d)
        return pd.Series(digits, dtype=int)

    def compute_distribution(self, digits: pd.Series) -> dict:
        if len(digits) == 0:
            return {d: 0.0 for d in range(1, 10)}
        counts = digits.value_counts().reindex(range(1, 10), fill_value=0)
        total  = len(digits)
        return {d: int(counts[d]) / total for d in range(1, 10)}

    def mad(self, observed: dict) -> float:
        """Mean Absolute Deviation from Benford's distribution."""
        return float(np.mean([abs(observed[d] - BENFORD[d]) for d in range(1, 10)]))

    def chi_square(self, observed: dict, n: int):
        obs_counts = np.array([observed[d] * n for d in range(1, 10)])
        exp_counts = BENFORD_ARR * n
        chi2, p = stats.chisquare(f_obs=obs_counts, f_exp=exp_counts)
        return float(chi2), float(p)

    def conformity_score(self, mad_val: float) -> float:
        """Convert MAD to 0–100 score (100 = perfect conformity)."""
        return round(max(0.0, 100.0 * (1.0 - mad_val / 0.05)), 1)

    def severity(self, mad_val: float, p: float) -> str:
        if mad_val <= MAD_THRESHOLDS["none"]:
            return "none"
        elif mad_val <= MAD_THRESHOLDS["low"]:
            return "low" if p > 0.01 else "medium"
        elif mad_val <= MAD_THRESHOLDS["medium"]:
            return "medium"
        elif mad_val <= 0.025:
            return "high"
        else:
            return "critical"

    def analyze(
        self,
        values: pd.Series,
        ticker: str = "UNKNOWN",
        concept: str = "figures",
    ) -> BenfordsResult:
        """
        Full Benford's Law analysis on a numeric series.

        Requires n ≥ 30 for reliable results (Nigrini recommendation).
        Still runs with fewer values but adds a warning.
        """
        digits   = self.extract_leading_digits(values.dropna())
        n        = len(digits)

        if n < 30:
            logger.warning(
                f"{ticker}/{concept}: n={n} < 30 — Benford's results unreliable"
            )

        observed = self.compute_distribution(digits)
        mad_val  = self.mad(observed)
        conf     = self.conformity_score(mad_val)
        chi2, p  = self.chi_square(observed, n) if n >= 5 else (0.0, 1.0)
        sev      = self.severity(mad_val, p)
        flagged  = sev in ("medium", "high", "critical")

        devs = sorted(
            range(1, 10),
            key=lambda d: abs(observed[d] - BENFORD[d]),
            reverse=True,
        )

        return BenfordsResult(
            ticker=ticker, concept=concept, n_values=n,
            observed=observed, expected=BENFORD.copy(),
            mad=mad_val, chi2_stat=chi2, chi2_p=p,
            conformity_score=conf, severity=sev, flagged=flagged,
            top_deviating=devs[:3],
            interpretation=self._interpret(
                ticker, concept, mad_val, p, sev, observed, devs[:2], n
            ),
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        ticker_col: str  = "ticker",
        value_col: str   = "value",
        concept_col: str = "concept",
    ) -> pd.DataFrame:
        """
        Run Benford's on each (ticker, concept) group.
        Returns a summary DataFrame sorted by MAD descending.
        """
        rows = []
        for (ticker, concept), grp in df.groupby([ticker_col, concept_col]):
            vals = pd.to_numeric(grp[value_col], errors="coerce").dropna()
            if len(vals) < 5:
                continue
            r = self.analyze(vals, str(ticker), str(concept))
            rows.append({
                "ticker":           r.ticker,
                "concept":          r.concept,
                "n_values":         r.n_values,
                "mad":              round(r.mad, 5),
                "conformity_score": r.conformity_score,
                "chi2_stat":        round(r.chi2_stat, 3),
                "chi2_p":           round(r.chi2_p, 4),
                "severity":         r.severity,
                "flagged":          r.flagged,
                "top_deviating":    str(r.top_deviating),
                "interpretation":   r.interpretation,
            })
        return pd.DataFrame(rows).sort_values("mad", ascending=False)

    def _interpret(self, ticker, concept, mad_val, p, sev, obs, top_devs, n) -> str:
        labels = {
            "none":     "✅ Close conformity",
            "low":      "⚠️ Minor deviation",
            "medium":   "🔶 Notable deviation",
            "high":     "🔴 Significant non-conformity",
            "critical": "🚨 Strong non-conformity",
        }
        base = f"{labels[sev]} — {concept} ({n} values), MAD={mad_val:.4f}"
        if sev == "none":
            return base + ". Distribution appears naturally occurring."

        desc = []
        for d in top_devs:
            obs_pct = obs[d] * 100
            exp_pct = BENFORD[d] * 100
            diff    = obs_pct - exp_pct
            word    = "over-represented" if diff > 0 else "under-represented"
            desc.append(f"Digit {d} {word} ({obs_pct:.1f}% vs {exp_pct:.1f}% expected)")

        detail = "; ".join(desc) + "."
        flag = (
            " Warrants forensic review — consistent with patterns in manipulated data."
            if sev in ("high", "critical") else
            " Cross-validate with Isolation Forest and YoY variance signals."
        )
        return f"{base}. {detail}{flag}"
