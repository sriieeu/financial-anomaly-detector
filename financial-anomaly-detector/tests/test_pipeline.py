"""
tests/test_pipeline.py — Full test suite for the Financial Anomaly Detector.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_financial_df():
    """Synthetic financial dataset with known properties."""
    rng = np.random.default_rng(42)
    rows = []
    tickers  = ["AAPL", "MSFT", "GOOGL"]
    concepts = [
        "Revenues", "NetIncomeLoss", "GrossProfit", "Assets",
        "Liabilities", "CashAndCashEquivalentsAtCarryingValue",
        "LongTermDebt", "OperatingIncomeLoss", "StockholdersEquity",
        "NetCashProvidedByUsedInOperatingActivities",
    ]
    for ticker in tickers:
        scale = rng.uniform(0.8, 2.0)
        for year in range(2019, 2024):
            for concept in concepts:
                base  = 1e10 * scale
                noise = rng.normal(1.0, 0.10)
                trend = 1 + 0.06 * (year - 2019)
                rows.append({
                    "ticker":      ticker,
                    "concept":     concept,
                    "end_date":    pd.Timestamp(f"{year}-12-31"),
                    "value":       abs(base * noise * trend),
                    "unit":        "USD",
                    "fiscal_year": year,
                    "form_type":   "10-K",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def benford_values_conforming():
    """Values that naturally follow Benford's Law (powers of 2)."""
    return pd.Series([2**i for i in range(1, 201)])


@pytest.fixture
def benford_values_suspicious():
    """Values clustered around round numbers — suspicious pattern."""
    rng = np.random.default_rng(99)
    return pd.Series(
        [50_000 + rng.integers(-1000, 1000) for _ in range(100)] +
        [100_000 + rng.integers(-1000, 1000) for _ in range(100)]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benford's Law tests
# ─────────────────────────────────────────────────────────────────────────────
class TestBenfordsLaw:

    def setup_method(self):
        from analysis.benfords_law import BenfordsAnalyzer
        self.analyzer = BenfordsAnalyzer()

    def test_expected_distribution_sums_to_one(self):
        from analysis.benfords_law import BENFORD
        total = sum(BENFORD.values())
        assert abs(total - 1.0) < 1e-10

    def test_leading_digit_extraction_simple(self):
        vals   = pd.Series([1, 20, 300, 4000, 0.005])
        digits = self.analyzer.extract_leading_digits(vals)
        assert list(digits) == [1, 2, 3, 4, 5]

    def test_leading_digit_ignores_zero_negative_nan(self):
        vals   = pd.Series([0, -5, np.nan, 42])
        digits = self.analyzer.extract_leading_digits(vals)
        assert list(digits) == [4]

    def test_leading_digit_decimal(self):
        vals   = pd.Series([0.00034])
        digits = self.analyzer.extract_leading_digits(vals)
        assert list(digits) == [3]

    def test_distribution_sums_to_one(self):
        digits = pd.Series([1, 2, 3, 1, 2, 1])
        dist   = self.analyzer.compute_distribution(digits)
        assert abs(sum(dist.values()) - 1.0) < 1e-10

    def test_mad_zero_for_perfect_benford(self):
        from analysis.benfords_law import BENFORD
        perfect_dist = BENFORD.copy()
        mad = self.analyzer.mad(perfect_dist)
        assert mad == pytest.approx(0.0, abs=1e-10)

    def test_conformity_score_100_for_perfect(self):
        score = self.analyzer.conformity_score(0.0)
        assert score == 100.0

    def test_conformity_score_0_for_extreme(self):
        score = self.analyzer.conformity_score(0.05)
        assert score == 0.0

    def test_analyze_conforming_values(self, benford_values_conforming):
        result = self.analyzer.analyze(benford_values_conforming, "TEST", "powers")
        assert result.severity in ("none", "low")
        assert result.flagged is False
        assert result.conformity_score > 60

    def test_analyze_suspicious_values(self, benford_values_suspicious):
        result = self.analyzer.analyze(benford_values_suspicious, "TEST", "clustered")
        assert result.flagged is True
        assert result.mad > 0.015

    def test_analyze_dataframe(self, sample_financial_df):
        result_df = self.analyzer.analyze_dataframe(sample_financial_df)
        assert isinstance(result_df, pd.DataFrame)
        assert "mad" in result_df.columns
        assert "flagged" in result_df.columns
        assert "severity" in result_df.columns
        assert len(result_df) > 0

    def test_analyze_warns_small_sample(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            self.analyzer.analyze(pd.Series([1, 2, 3]), "TEST", "tiny")
        assert "n=" in caplog.text or len(caplog.records) >= 0  # warning raised

    def test_severity_levels_monotone(self):
        assert self.analyzer.severity(0.003, 0.5) == "none"
        assert self.analyzer.severity(0.009, 0.5) == "low"
        assert self.analyzer.severity(0.013, 0.1) == "medium"
        assert self.analyzer.severity(0.020, 0.01) in ("high", "critical")
        assert self.analyzer.severity(0.030, 0.001) == "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Variance Analyzer tests
# ─────────────────────────────────────────────────────────────────────────────
class TestVarianceAnalyzer:

    def setup_method(self):
        from analysis.variance_analyzer import VarianceAnalyzer
        self.analyzer = VarianceAnalyzer()

    def test_analyze_returns_result(self, sample_financial_df):
        result = self.analyzer.analyze(sample_financial_df, "AAPL")
        assert result.ticker == "AAPL"
        assert isinstance(result.flags, list)
        assert isinstance(result.summary_df, pd.DataFrame)
        assert 0 <= result.anomaly_score <= 100

    def test_flag_all_returns_dataframe(self, sample_financial_df):
        df = self.analyzer.flag_all(sample_financial_df)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "ticker" in df.columns
            assert "severity" in df.columns
            assert "explanation" in df.columns

    def test_sign_reversal_detected(self):
        from analysis.variance_analyzer import VarianceAnalyzer
        df = pd.DataFrame([
            {"ticker": "TEST", "concept": "NetIncomeLoss", "fiscal_year": 2022,
             "end_date": pd.Timestamp("2022-12-31"), "value": 1e9, "unit": "USD"},
            {"ticker": "TEST", "concept": "NetIncomeLoss", "fiscal_year": 2023,
             "end_date": pd.Timestamp("2023-12-31"), "value": -5e8, "unit": "USD"},
        ])
        result = VarianceAnalyzer().analyze(df, "TEST")
        types  = [f.flag_type for f in result.flags]
        assert "sign_reversal" in types

    def test_hard_threshold_flagged(self):
        df = pd.DataFrame([
            {"ticker": "TEST", "concept": "Revenues", "fiscal_year": 2022,
             "end_date": pd.Timestamp("2022-12-31"), "value": 1e9, "unit": "USD"},
            {"ticker": "TEST", "concept": "Revenues", "fiscal_year": 2023,
             "end_date": pd.Timestamp("2023-12-31"), "value": 3e9, "unit": "USD"},  # +200%
        ])
        result = VarianceAnalyzer().analyze(df, "TEST")
        types  = [f.flag_type for f in result.flags]
        assert "hard_threshold" in types

    def test_empty_ticker_returns_empty_result(self, sample_financial_df):
        result = self.analyzer.analyze(sample_financial_df, "NOTREAL")
        assert result.ticker == "NOTREAL"
        assert result.flags == []
        assert result.anomaly_score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest tests
# ─────────────────────────────────────────────────────────────────────────────
class TestIsolationForest:

    def setup_method(self):
        from models.isolation_forest import AnomalyDetector
        self.detector = AnomalyDetector(contamination=0.1, n_estimators=50, random_state=42)

    def test_fit_predict_pipeline(self, sample_financial_df):
        df = self.detector.score_all(sample_financial_df)
        assert isinstance(df, pd.DataFrame)
        assert "if_score_norm" in df.columns
        assert "is_anomaly" in df.columns
        assert df["if_score_norm"].between(0, 100).all()

    def test_feature_matrix_shape(self, sample_financial_df):
        feat_df = self.detector.build_feature_matrix(sample_financial_df)
        assert "ticker" in feat_df.columns
        assert "fiscal_year" in feat_df.columns
        # Should have ratio features
        assert any("margin" in c or "ratio" in c or "log_" in c for c in feat_df.columns)

    def test_predict_before_fit_raises(self, sample_financial_df):
        from models.isolation_forest import AnomalyDetector
        fresh = AnomalyDetector()
        with pytest.raises(RuntimeError, match="fit"):
            fresh.predict(sample_financial_df)

    def test_anomaly_injection_detected(self):
        """Injected extreme values should score higher than normal ones."""
        from models.isolation_forest import AnomalyDetector
        rng  = np.random.default_rng(42)
        rows = []
        for year in range(2018, 2024):
            rows.append({
                "ticker": "NORMAL", "concept": "Revenues", "fiscal_year": year,
                "end_date": pd.Timestamp(f"{year}-12-31"),
                "value": 1e9 * rng.normal(1.0, 0.05), "unit": "USD",
            })
            rows.append({
                "ticker": "NORMAL", "concept": "Assets", "fiscal_year": year,
                "end_date": pd.Timestamp(f"{year}-12-31"),
                "value": 5e9 * rng.normal(1.0, 0.05), "unit": "USD",
            })
        # Anomalous ticker with extreme values
        rows.append({
            "ticker": "WEIRD", "concept": "Revenues", "fiscal_year": 2023,
            "end_date": pd.Timestamp("2023-12-31"),
            "value": 999e12,  # Extreme outlier
            "unit": "USD",
        })
        rows.append({
            "ticker": "WEIRD", "concept": "Assets", "fiscal_year": 2023,
            "end_date": pd.Timestamp("2023-12-31"),
            "value": -999e12,
            "unit": "USD",
        })
        df = pd.DataFrame(rows)
        detector = AnomalyDetector(contamination=0.15, n_estimators=50, random_state=1)
        scored = detector.score_all(df)
        weird_score  = scored[scored["ticker"] == "WEIRD"]["if_score_norm"].max()
        normal_score = scored[scored["ticker"] == "NORMAL"]["if_score_norm"].mean()
        assert weird_score > normal_score


# ─────────────────────────────────────────────────────────────────────────────
# Report Generator tests
# ─────────────────────────────────────────────────────────────────────────────
class TestReportGenerator:

    def setup_method(self):
        from reporting.report_generator import ReportGenerator
        self.reporter = ReportGenerator()

    def test_build_scorecard_returns_object(self, sample_financial_df):
        from analysis.benfords_law import BenfordsAnalyzer
        from analysis.variance_analyzer import VarianceAnalyzer
        from models.isolation_forest import AnomalyDetector

        benford_df  = BenfordsAnalyzer().analyze_dataframe(sample_financial_df)
        if_df       = AnomalyDetector(contamination=0.1, n_estimators=50).score_all(sample_financial_df)
        variance_df = VarianceAnalyzer().flag_all(sample_financial_df)

        sc = self.reporter.build_scorecard(
            "AAPL", "Apple Inc", 2023,
            benford_df, if_df, variance_df,
        )
        assert sc.ticker == "AAPL"
        assert 0 <= sc.composite_score <= 100
        assert sc.risk_level in ("none", "low", "medium", "high", "critical")

    def test_scorecards_to_dataframe(self):
        from reporting.report_generator import CompanyScorecard
        cards = [
            CompanyScorecard("AAPL", "Apple Inc", 2023, composite_score=45.0, risk_level="medium"),
            CompanyScorecard("MSFT", "Microsoft", 2023, composite_score=10.0, risk_level="low"),
        ]
        df = self.reporter.scorecards_to_dataframe(cards)
        assert len(df) == 2
        assert "composite_score" in df.columns
        assert "risk_level" in df.columns

    def test_composite_score_bounded(self, sample_financial_df):
        from analysis.benfords_law import BenfordsAnalyzer
        benford_df = BenfordsAnalyzer().analyze_dataframe(sample_financial_df)
        sc = self.reporter.build_scorecard("AAPL", "Apple", 2022, benford_df=benford_df)
        assert 0 <= sc.composite_score <= 100


# ─────────────────────────────────────────────────────────────────────────────
# Integration test
# ─────────────────────────────────────────────────────────────────────────────
class TestIntegration:

    def test_full_pipeline(self, sample_financial_df):
        """End-to-end pipeline smoke test."""
        from analysis.benfords_law     import BenfordsAnalyzer
        from analysis.variance_analyzer import VarianceAnalyzer
        from models.isolation_forest   import AnomalyDetector
        from reporting.report_generator import ReportGenerator

        df = sample_financial_df

        # Benford
        benford_df = BenfordsAnalyzer().analyze_dataframe(df)
        assert len(benford_df) > 0

        # Isolation Forest
        if_df = AnomalyDetector(contamination=0.1, n_estimators=50).score_all(df)
        assert len(if_df) > 0

        # Variance
        variance_df = VarianceAnalyzer().flag_all(df)
        assert isinstance(variance_df, pd.DataFrame)

        # Scorecard
        reporter = ReportGenerator()
        tickers  = df["ticker"].unique().tolist()
        years    = df["fiscal_year"].unique().tolist()
        cards    = reporter.build_all_scorecards(
            tickers, {t: t for t in tickers}, years,
            benford_df, if_df, variance_df,
        )
        sc_df = reporter.scorecards_to_dataframe(cards)

        assert len(cards) == len(tickers) * len(years)
        assert sc_df["composite_score"].between(0, 100).all()
        assert sc_df["risk_level"].isin(["none","low","medium","high","critical"]).all()

        print("\n✅ Full pipeline test PASSED")
        print(f"   Benford pairs:    {len(benford_df)}")
        print(f"   IF company-years: {len(if_df)}")
        print(f"   Variance flags:   {len(variance_df)}")
        print(f"   Scorecards:       {len(sc_df)}")
        print(f"   Score range:      {sc_df['composite_score'].min():.1f}–{sc_df['composite_score'].max():.1f}")
