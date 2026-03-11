"""
pdf_parser.py — Extracts financial tables from SEC 10-K/10-Q PDFs using pdfplumber.

Strategy:
1. Scan pages for financial statement keywords
2. Extract tables with pdfplumber (lattice strategy for ruled tables)
3. Detect statement type: income_statement / balance_sheet / cash_flow
4. Clean cells: parse (1,234) as -1234, strip footnotes, normalise numbers

Usage:
    parser = PDFParser()
    tables = parser.extract_from_bytes(pdf_bytes)
    for t in tables:
        print(t.statement_type, t.df.shape)
"""

import io
import logging
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

INCOME_KW   = ["revenue", "net income", "gross profit", "operating income", "cost of revenue", "earnings per share"]
BALANCE_KW  = ["total assets", "total liabilities", "stockholders equity", "cash and cash equivalents", "long-term debt"]
CASHFLOW_KW = ["operating activities", "investing activities", "financing activities", "capital expenditures"]

STATEMENT_KEYWORDS = {
    "income_statement": INCOME_KW,
    "balance_sheet":    BALANCE_KW,
    "cash_flow":        CASHFLOW_KW,
}

PAREN_RE    = re.compile(r"^\(([0-9,\.]+)\)$")
NUMBER_RE   = re.compile(r"^-?[0-9,\.]+$")
FOOTNOTE_RE = re.compile(r"^\(\d+\)$")
YEAR_RE     = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class FinancialTable:
    statement_type: str
    page_number: int
    df: pd.DataFrame
    years_detected: list
    confidence: float


class PDFParser:
    """Extract financial tables from SEC PDF/HTML filings."""

    def extract_from_bytes(self, pdf_bytes: bytes) -> list:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber required: pip install pdfplumber")

        tables = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages[:200], start=1):
                text = (page.extract_text() or "").lower()
                if not self._is_financial_page(text):
                    continue

                stmt_type  = self._detect_type(text)
                raw_tables = page.extract_tables({
                    "vertical_strategy":   "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance":      5,
                }) or []

                for raw in raw_tables:
                    ft = self._process(raw, stmt_type, page_num)
                    if ft:
                        tables.append(ft)

        return sorted(tables, key=lambda x: x.confidence, reverse=True)

    def extract_from_html(self, html: str) -> list:
        try:
            raw_tables = pd.read_html(io.StringIO(html), thousands=",", displayed_only=False)
        except Exception as e:
            logger.warning(f"HTML table parse failed: {e}")
            return []

        tables = []
        for i, df in enumerate(raw_tables):
            if df.shape[0] < 3 or df.shape[1] < 2:
                continue
            text = " ".join(str(v) for v in df.values.flatten()).lower()
            if not self._is_financial_page(text):
                continue
            stmt_type = self._detect_type(text)
            df_clean  = self._clean_df(df)
            years     = self._years(df)
            conf      = self._confidence(df_clean, stmt_type)
            if conf > 0.2:
                tables.append(FinancialTable(stmt_type, i, df_clean, years, conf))

        return sorted(tables, key=lambda x: x.confidence, reverse=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_financial_page(self, text: str) -> bool:
        all_kw = INCOME_KW + BALANCE_KW + CASHFLOW_KW
        return sum(1 for kw in all_kw if kw in text) >= 2

    def _detect_type(self, text: str) -> str:
        scores = {k: sum(1 for kw in v if kw in text) for k, v in STATEMENT_KEYWORDS.items()}
        return max(scores, key=scores.get) if any(scores.values()) else "unknown"

    def _process(self, raw, stmt_type, page_num) -> Optional[FinancialTable]:
        if not raw or len(raw) < 3:
            return None
        try:
            df = pd.DataFrame(raw[1:], columns=raw[0])
        except Exception:
            return None
        df_clean = self._clean_df(df)
        if df_clean.empty or df_clean.shape[1] < 2:
            return None
        years = self._years(df)
        conf  = self._confidence(df_clean, stmt_type)
        return FinancialTable(stmt_type, page_num, df_clean, years, conf) if conf >= 0.15 else None

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().dropna(how="all").dropna(axis=1, how="all")
        df.columns = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(df.columns)]
        for col in df.columns:
            df[col] = df[col].apply(self._clean_cell)
        return df.dropna(how="all").reset_index(drop=True)

    def _clean_cell(self, val):
        if val is None:
            return None
        s = str(val).strip()
        if not s or s in ("-", "—", "–", "N/A"):
            return None
        if FOOTNOTE_RE.match(s):
            return None
        m = PAREN_RE.match(s)
        if m:
            try:
                return -float(m.group(1).replace(",", ""))
            except ValueError:
                pass
        clean = s.replace(",", "").replace("$", "").replace("%", "").strip()
        if NUMBER_RE.match(clean):
            try:
                return float(clean)
            except ValueError:
                pass
        return s

    def _years(self, df: pd.DataFrame) -> list:
        years = []
        for col in df.columns:
            for m in YEAR_RE.finditer(str(col)):
                years.append(int(m.group()))
        return sorted(set(years), reverse=True)

    def _confidence(self, df: pd.DataFrame, stmt_type: str) -> float:
        if df.empty:
            return 0.0
        num = sum(isinstance(v, (int, float)) for col in df.columns[1:] for v in df[col])
        tot = sum(1 for col in df.columns[1:] for _ in df[col])
        ratio = num / tot if tot else 0
        bonus = 0.2 if stmt_type != "unknown" else 0
        return min(1.0, ratio * 0.6 + bonus + min(0.2, (df.shape[1] - 1) * 0.05))
