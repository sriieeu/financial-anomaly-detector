"""
isolation_forest.py — Isolation Forest for multivariate financial anomaly detection.

Isolation Forest isolates anomalies by randomly partitioning the feature space.
Anomalies require fewer partitions to isolate (shorter path length) and receive
lower anomaly scores closer to -1.

Features used:
- Financial ratios (P/E proxy, debt ratios, liquidity ratios)
- YoY change rates for key line items
- Accrual-based metrics
- Cross-sectional z-scores vs peers

Usage:
    detector = AnomalyDetector()
    detector.fit(train_df)
    results  = detector.predict(test_df)
    df_out   = detector.score_all(full_df)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# Features to build for anomaly detection
RATIO_FEATURES = [
    # Profitability
    ("NetIncomeLoss",          "Revenues",           "net_profit_margin"),
    ("GrossProfit",            "Revenues",           "gross_margin"),
    ("OperatingIncomeLoss",    "Revenues",           "operating_margin"),
    # Leverage
    ("Liabilities",            "Assets",             "debt_to_assets"),
    ("LongTermDebt",           "Assets",             "lt_debt_ratio"),
    # Liquidity
    ("AssetsCurrent",          "LiabilitiesCurrent", "current_ratio"),
    ("CashAndCashEquivalentsAtCarryingValue", "LiabilitiesCurrent", "cash_ratio"),
    # Cash quality
    ("NetCashProvidedByUsedInOperatingActivities", "NetIncomeLoss", "cash_earnings_ratio"),
    # Growth
    ("ResearchAndDevelopmentExpense", "Revenues",    "r_and_d_intensity"),
]


@dataclass
class AnomalyResult:
    ticker: str
    fiscal_year: int
    anomaly_score: float        # Sklearn raw: -1 (anomaly) to 1 (normal)
    normalized_score: float     # 0–100 (100 = most anomalous)
    is_anomaly: bool
    contamination: float        # Threshold used
    top_features: list          # Features that contributed most
    explanation: str


class AnomalyDetector:
    """
    Isolation Forest–based anomaly detector for financial statements.

    Trains on cross-sectional multi-year data, scores each
    (ticker, fiscal_year) observation, flags statistical outliers.

    The contamination parameter controls the expected fraction of
    anomalies in the training data. Default 0.05 (5%).
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            max_features=1.0,
        )
        self.scaler        = RobustScaler()
        self.feature_names = []
        self._fitted       = False

    def build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot financial data into a feature matrix:
        rows = (ticker, fiscal_year), cols = financial features.

        Features include:
        - Raw concept values (log-transformed for scale invariance)
        - Financial ratios (e.g. net margin, debt-to-assets)
        - YoY growth rates
        """
        # Pivot: index=(ticker, fiscal_year), columns=concept, values=value
        pivot = df.pivot_table(
            index=["ticker", "fiscal_year"],
            columns="concept",
            values="value",
            aggfunc="last",
        ).reset_index()

        features = pd.DataFrame()
        features["ticker"]      = pivot["ticker"]
        features["fiscal_year"] = pivot["fiscal_year"]

        # ── Raw log values ──
        raw_cols = [c for c in pivot.columns if c not in ("ticker", "fiscal_year")]
        for col in raw_cols:
            if col in pivot.columns:
                vals = pd.to_numeric(pivot[col], errors="coerce")
                # Log transform (handle negatives with sign-preserving log)
                features[f"log_{col}"] = np.sign(vals) * np.log1p(np.abs(vals))

        # ── Financial ratios ──
        for num_concept, den_concept, feat_name in RATIO_FEATURES:
            if num_concept in pivot.columns and den_concept in pivot.columns:
                num = pd.to_numeric(pivot[num_concept], errors="coerce")
                den = pd.to_numeric(pivot[den_concept], errors="coerce")
                # Safe division — zero denominator → NaN
                ratio = np.where(
                    (den.notna()) & (den.abs() > 1e-6),
                    num / den,
                    np.nan,
                )
                features[feat_name] = ratio

        # ── YoY growth rates ──
        growth_concepts = [
            "Revenues", "NetIncomeLoss", "Assets",
            "CashAndCashEquivalentsAtCarryingValue",
        ]
        for concept in growth_concepts:
            if concept in pivot.columns:
                vals = pd.to_numeric(pivot[concept], errors="coerce")
                # Sort by fiscal year within each ticker, then pct_change
                tmp = pd.DataFrame({
                    "ticker": features["ticker"],
                    "fy":     features["fiscal_year"],
                    "val":    vals,
                }).sort_values(["ticker", "fy"])
                tmp["growth"] = tmp.groupby("ticker")["val"].pct_change()
                tmp["growth"] = tmp["growth"].clip(-5, 5)  # Clip extreme values
                features[f"yoy_{concept}"] = tmp["growth"].values

        return features

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit Isolation Forest on financial feature matrix.

        Args:
            df: Full financial dataset (multiple tickers, multiple years)
        """
        feat_df = self.build_feature_matrix(df)
        self.feature_names = [
            c for c in feat_df.columns
            if c not in ("ticker", "fiscal_year")
        ]

        X = feat_df[self.feature_names].copy()

        # Impute NaN with column median (most financial NaN = concept not reported)
        self._medians = X.median()
        X = X.fillna(self._medians)

        # Robust scaling (RobustScaler ignores outliers in scaling params)
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self._fitted       = True
        self._feat_df_cols = list(feat_df.columns)
        logger.info(
            f"Isolation Forest fitted on {len(X)} samples × {len(self.feature_names)} features"
        )
        return self

    def predict(self, df: pd.DataFrame) -> list:
        """
        Score a dataset. Returns list of AnomalyResult.

        Can call predict() on new data after fit() on training data.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        feat_df = self.build_feature_matrix(df)
        X = feat_df[self.feature_names].fillna(self._medians)
        X_scaled = self.scaler.transform(X)

        # Raw scores: negative = more anomalous (sklearn convention)
        raw_scores = self.model.score_samples(X_scaled)   # -1 to 0 range
        labels     = self.model.predict(X_scaled)          # -1 anomaly, 1 normal

        # Normalise to 0–100 (100 = most anomalous)
        # score_samples returns avg path length; more negative = shorter path = anomaly
        norm = 100 * (1 - (raw_scores - raw_scores.min()) / (np.ptp(raw_scores) + 1e-9))

        results = []
        for i, (_, row) in enumerate(feat_df.iterrows()):
            # Top deviating features for explanation
            feat_vals = X.iloc[i]
            z_scores  = ((feat_vals - self._medians) / (X.std() + 1e-9)).abs()
            top_feats = z_scores.nlargest(3).index.tolist()

            results.append(AnomalyResult(
                ticker=str(row["ticker"]),
                fiscal_year=int(row["fiscal_year"]),
                anomaly_score=float(raw_scores[i]),
                normalized_score=float(norm[i]),
                is_anomaly=bool(labels[i] == -1),
                contamination=self.contamination,
                top_features=top_feats,
                explanation=self._explain(
                    str(row["ticker"]), int(row["fiscal_year"]),
                    float(norm[i]), top_feats, bool(labels[i] == -1)
                ),
            ))

        return results

    def score_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit + predict on the same dataset. Returns a scored DataFrame.
        """
        self.fit(df)
        results = self.predict(df)
        return pd.DataFrame([{
            "ticker":           r.ticker,
            "fiscal_year":      r.fiscal_year,
            "if_score_raw":     round(r.anomaly_score, 4),
            "if_score_norm":    round(r.normalized_score, 1),
            "is_anomaly":       r.is_anomaly,
            "top_features":     ", ".join(r.top_features),
            "explanation":      r.explanation,
        } for r in results]).sort_values("if_score_norm", ascending=False)

    def _explain(
        self, ticker: str, year: int, score: float,
        top_feats: list, is_anomaly: bool,
    ) -> str:
        label = "🚨 Anomaly detected" if is_anomaly else "✅ Normal"
        feat_desc = ", ".join(f.replace("_", " ").replace("log ", "") for f in top_feats[:3])
        return (
            f"{label} — {ticker} FY{year} (score: {score:.0f}/100). "
            f"Primary deviating features: {feat_desc}."
            + (
                " Financial profile is a statistical outlier vs peer universe."
                if is_anomaly else ""
            )
        )
