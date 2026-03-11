"""
edgar_client.py — SEC EDGAR API client.

Fetches 10-K / 10-Q filings via the official EDGAR data APIs:
  - Ticker → CIK:    https://data.sec.gov/files/company_tickers.json
  - Submissions:     https://data.sec.gov/submissions/CIK{cik}.json
  - XBRL facts:      https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

EDGAR fair-use: max 10 req/s, User-Agent header required.

Usage:
    client   = EDGARClient()
    df       = client.get_financial_dataframe("AAPL", form_type="10-K")
    filings  = client.get_filings("MSFT", form_types=["10-K"], years=5)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL            = "https://data.sec.gov"
SUBMISSIONS_URL     = f"{BASE_URL}/submissions/CIK{{cik}}.json"
COMPANY_FACTS_URL   = f"{BASE_URL}/api/xbrl/companyfacts/CIK{{cik}}.json"
COMPANY_TICKERS_URL = f"{BASE_URL}/files/company_tickers.json"
RATE_LIMIT_DELAY    = 0.15  # 6.7 req/s — well within 10 req/s limit

# Core financial concepts to extract from XBRL
KEY_CONCEPTS = [
    "us-gaap/Revenues",
    "us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap/NetIncomeLoss",
    "us-gaap/GrossProfit",
    "us-gaap/OperatingIncomeLoss",
    "us-gaap/Assets",
    "us-gaap/AssetsCurrent",
    "us-gaap/Liabilities",
    "us-gaap/LiabilitiesCurrent",
    "us-gaap/StockholdersEquity",
    "us-gaap/CashAndCashEquivalentsAtCarryingValue",
    "us-gaap/LongTermDebt",
    "us-gaap/RetainedEarningsAccumulatedDeficit",
    "us-gaap/OperatingExpenses",
    "us-gaap/CostOfRevenue",
    "us-gaap/ResearchAndDevelopmentExpense",
    "us-gaap/SellingGeneralAndAdministrativeExpense",
    "us-gaap/EarningsPerShareBasic",
    "us-gaap/CommonStockSharesOutstanding",
    "us-gaap/NetCashProvidedByUsedInOperatingActivities",
    "us-gaap/NetCashProvidedByUsedInInvestingActivities",
    "us-gaap/NetCashProvidedByUsedInFinancingActivities",
    "us-gaap/DepreciationDepletionAndAmortization",
    "us-gaap/CapitalExpendituresIncurredButNotYetPaid",
    "us-gaap/AccountsReceivableNetCurrent",
    "us-gaap/InventoryNet",
    "us-gaap/PropertyPlantAndEquipmentNet",
    "us-gaap/GoodwillAndIntangibleAssetsNetOfAmortization",
    "us-gaap/IncomeTaxExpenseBenefit",
]


@dataclass
class FilingMetadata:
    cik: str
    ticker: str
    company_name: str
    form_type: str
    filed_date: str
    period_of_report: str
    accession_number: str
    primary_document: str
    filing_url: str
    fiscal_year: Optional[int] = None


@dataclass
class CompanyFacts:
    cik: str
    ticker: str
    company_name: str
    facts: dict = field(default_factory=dict)


class EDGARClient:
    """
    Official SEC EDGAR API client with rate limiting and caching.

    Extracts structured XBRL financial data — no scraping needed.
    EDGAR's XBRL database contains structured financials for all
    public companies since ~2009.
    """

    def __init__(
        self,
        user_agent: str = "Financial Anomaly Detector research@example.com",
    ):
        self.user_agent = user_agent
        self.session    = requests.Session()
        self.session.headers.update({
            "User-Agent":      user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._ticker_map: Optional[dict] = None
        self._last_req   = 0.0

    # ── HTTP ──────────────────────────────────────────────────────────────────

    def _get(self, url: str, host: str = "data.sec.gov") -> dict:
        elapsed = time.time() - self._last_req
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.session.headers["Host"] = host
        logger.debug(f"GET {url}")
        r = self.session.get(url, timeout=30)
        self._last_req = time.time()
        r.raise_for_status()
        return r.json()

    # ── Ticker / CIK lookup ───────────────────────────────────────────────────

    def _load_ticker_map(self) -> dict:
        if self._ticker_map is not None:
            return self._ticker_map
        logger.info("Loading EDGAR ticker→CIK map…")
        data = self._get(COMPANY_TICKERS_URL)
        self._ticker_map = {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
        }
        logger.info(f"Loaded {len(self._ticker_map):,} tickers")
        return self._ticker_map

    def ticker_to_cik(self, ticker: str) -> str:
        m = self._load_ticker_map()
        t = ticker.upper().strip()
        if t not in m:
            raise ValueError(f"Ticker '{t}' not found in EDGAR")
        return m[t]

    # ── Filings ───────────────────────────────────────────────────────────────

    def get_filings(
        self,
        ticker: str,
        form_types: Optional[list] = None,
        years: int = 5,
        cik: Optional[str] = None,
    ) -> list:
        import datetime
        form_types = form_types or ["10-K", "10-Q"]
        cik        = cik or self.ticker_to_cik(ticker)
        data       = self._get(SUBMISSIONS_URL.format(cik=cik))

        company_name = data.get("name", ticker)
        recent       = data.get("filings", {}).get("recent", {})
        cutoff_year  = datetime.datetime.now().year - years

        results = []
        for form, date, period, acc, primary in zip(
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("reportDate", []),
            recent.get("accessionNumber", []),
            recent.get("primaryDocument", []),
        ):
            if form not in form_types:
                continue
            try:
                if int(date[:4]) < cutoff_year:
                    continue
            except (ValueError, TypeError):
                continue

            acc_clean = acc.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/{primary}"
            )
            results.append(FilingMetadata(
                cik=cik, ticker=ticker.upper(),
                company_name=company_name, form_type=form,
                filed_date=date, period_of_report=period,
                accession_number=acc, primary_document=primary,
                filing_url=url, fiscal_year=int(period[:4]) if period else None,
            ))

        return sorted(results, key=lambda x: x.filed_date, reverse=True)

    # ── XBRL Financial Facts ──────────────────────────────────────────────────

    def get_company_facts(
        self, ticker: str, cik: Optional[str] = None
    ) -> CompanyFacts:
        cik  = cik or self.ticker_to_cik(ticker)
        data = self._get(COMPANY_FACTS_URL.format(cik=cik))

        company_name = data.get("entityName", ticker)
        raw_facts    = data.get("facts", {})

        flat = {}
        for taxonomy, concepts in raw_facts.items():
            for concept, meta in concepts.items():
                for unit_type, entries in meta.get("units", {}).items():
                    key = f"{taxonomy}/{concept}"
                    rows = [
                        {
                            "end":   e.get("end"),
                            "val":   e.get("val"),
                            "form":  e.get("form"),
                            "frame": e.get("frame"),
                            "unit":  unit_type,
                        }
                        for e in entries
                        if e.get("form") in ("10-K", "10-Q")
                        and e.get("val") is not None
                    ]
                    if rows:
                        flat[key] = rows

        logger.info(
            f"Loaded {len(flat):,} XBRL concepts for {ticker} ({company_name})"
        )
        return CompanyFacts(
            cik=cik, ticker=ticker.upper(),
            company_name=company_name, facts=flat,
        )

    def get_financial_dataframe(
        self,
        ticker: str,
        concepts: Optional[list] = None,
        form_type: str = "10-K",
        cik: Optional[str] = None,
    ):
        """
        Return a tidy DataFrame of financial figures for a company.

        Columns: ticker, concept, end_date, value, unit, fiscal_year
        """
        import pandas as pd

        concepts = concepts or KEY_CONCEPTS
        facts    = self.get_company_facts(ticker, cik=cik)

        rows = []
        for concept in concepts:
            for entry in facts.facts.get(concept, []):
                if entry.get("form") != form_type:
                    continue
                rows.append({
                    "ticker":    ticker.upper(),
                    "concept":   concept.split("/")[-1],
                    "end_date":  entry["end"],
                    "value":     entry["val"],
                    "unit":      entry["unit"],
                })

        if not rows:
            logger.warning(f"No {form_type} data for {ticker}")
            return pd.DataFrame()

        df             = pd.DataFrame(rows)
        df["end_date"] = pd.to_datetime(df["end_date"])
        df["fiscal_year"] = df["end_date"].dt.year
        return df.sort_values(["concept", "end_date"]).reset_index(drop=True)
