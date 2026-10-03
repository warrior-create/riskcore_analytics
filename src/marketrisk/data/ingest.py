"""
Data ingestion layer for MarketRisk-Lab.

Downloads and stores:
  - Equity adjusted closes (yfinance)
  - India VIX (yfinance)
  - Nifty 50 index (yfinance)
  - USDINR spot (yfinance)
  - INR yield curve (synthetic 3-point from FRED USD rates + spread)

All data is stored in the SQLite database defined in schema.py.
A SHA-256 hash of each dataset is recorded in dataset_hashes.

Usage
-----
    from marketrisk.data.ingest import DataIngestor
    ingestor = DataIngestor(cfg)
    ingestor.run_full()
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from marketrisk.data.schema import get_connection

logger = logging.getLogger(__name__)


def _sha256_df(df: pd.DataFrame) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    return hashlib.sha256(buf.getvalue()).hexdigest()


class DataIngestor:
    """
    Downloads all required market data and populates the SQLite DB.

    Parameters
    ----------
    cfg : dict loaded from config/settings.yaml
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.db_path = Path(cfg["data"]["db_path"])
        self.conn = get_connection(self.db_path)
        self._fred_key = (
            cfg["data"].get("fred_api_key") or os.environ.get("FRED_API_KEY")
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_full(self, start: Optional[str] = None, end: Optional[str] = None) -> None:
        """Download everything: equities, indices, FX, yield curve."""
        start = start or self.cfg["data"]["history_start"]
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

        logger.info("=== DataIngestor.run_full: %s → %s ===", start, end)

        tickers_equity = self.cfg["data"]["equity_tickers"] + [
            self.cfg["data"]["ipo_ticker"],
            self.cfg["data"]["nifty_ticker"],
        ]

        self._ingest_prices(tickers_equity, start, end)
        self._ingest_vol_index(start, end)
        self._ingest_fx(start, end)
        self._ingest_yield_curve(start, end)

        logger.info("=== DataIngestor.run_full: COMPLETE ===")

    # ------------------------------------------------------------------
    # Equities
    # ------------------------------------------------------------------

    def _ingest_prices(self, tickers: list[str], start: str, end: str) -> None:
        logger.info("Downloading prices for %d tickers …", len(tickers))
        for ticker in tickers:
            try:
                self._download_single_equity(ticker, start, end)
                time.sleep(0.3)   # be polite to yfinance
            except Exception as exc:
                logger.error("Failed to download %s: %s", ticker, exc)

    def _download_single_equity(self, ticker: str, start: str, end: str) -> None:
        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )
        if raw.empty:
            logger.warning("Empty data for %s", ticker)
            return

        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ["_".join(c).strip("_") for c in raw.columns]
            # Rename to standard names
            col_map = {}
            for c in raw.columns:
                for standard in ["Open", "High", "Low", "Close", "Volume"]:
                    if standard.lower() in c.lower():
                        col_map[c] = standard
            raw = raw.rename(columns=col_map)

        # auto_adjust=True means Close IS the adjusted close
        raw = raw.rename(columns={"Close": "adj_close"})
        if "Open" not in raw.columns:
            raw["Open"] = np.nan
        if "High" not in raw.columns:
            raw["High"] = np.nan
        if "Low" not in raw.columns:
            raw["Low"] = np.nan
        if "Volume" not in raw.columns:
            raw["Volume"] = np.nan

        rows = []
        for dt, row in raw.iterrows():
            rows.append(
                (
                    pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    ticker,
                    _safe_float(row.get("Open")),
                    _safe_float(row.get("High")),
                    _safe_float(row.get("Low")),
                    _safe_float(row.get("adj_close")),
                    _safe_float(row.get("adj_close")),   # adj_close again
                    _safe_int(row.get("Volume")),
                    "yfinance",
                )
            )

        cur = self.conn.cursor()
        cur.executemany(
            """INSERT OR REPLACE INTO prices
               (date, ticker, open, high, low, close, adj_close, volume, source)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()

        # Record hash
        self._record_hash(f"prices_{ticker}", raw, len(rows))
        logger.info("  %s: %d rows inserted/updated", ticker, len(rows))

    # ------------------------------------------------------------------
    # VIX
    # ------------------------------------------------------------------

    def _ingest_vol_index(self, start: str, end: str) -> None:
        vix_ticker = self.cfg["data"]["vix_ticker"]
        logger.info("Downloading India VIX (%s) …", vix_ticker)
        try:
            raw = yf.download(vix_ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                logger.warning("Empty VIX data")
                return

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = ["_".join(c).strip("_") for c in raw.columns]

            # Use Close column
            close_col = next(
                (c for c in raw.columns if "close" in c.lower()), None
            )
            if close_col is None:
                logger.warning("No close column in VIX data")
                return

            rows = [
                (
                    pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "INDIAVIX",
                    float(raw.loc[dt, close_col]),
                    "yfinance",
                )
                for dt in raw.index
                if not pd.isna(raw.loc[dt, close_col])
            ]
            cur = self.conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO vol_index (date, index_name, value, source) VALUES (?,?,?,?)",
                rows,
            )
            self.conn.commit()
            self._record_hash("vol_index_INDIAVIX", raw, len(rows))
            logger.info("  VIX: %d rows", len(rows))
        except Exception as exc:
            logger.error("VIX download failed: %s", exc)

    # ------------------------------------------------------------------
    # FX
    # ------------------------------------------------------------------

    def _ingest_fx(self, start: str, end: str) -> None:
        fx_ticker = self.cfg["data"]["usdinr_ticker"]
        logger.info("Downloading USDINR (%s) …", fx_ticker)
        try:
            raw = yf.download(fx_ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                logger.warning("Empty USDINR data")
                return

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = ["_".join(c).strip("_") for c in raw.columns]

            close_col = next((c for c in raw.columns if "close" in c.lower()), None)
            if close_col is None:
                return

            rows = []
            for dt in raw.index:
                val = raw.loc[dt, close_col]
                if pd.isna(val):
                    continue
                rows.append(
                    (pd.Timestamp(dt).strftime("%Y-%m-%d"), "USDINR", float(val), "yfinance")
                )

            cur = self.conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO fx_rates (date, ccy_pair, rate, source) VALUES (?,?,?,?)",
                rows,
            )
            self.conn.commit()
            self._record_hash("fx_USDINR", raw, len(rows))
            logger.info("  USDINR: %d rows", len(rows))
        except Exception as exc:
            logger.error("FX download failed: %s", exc)

    # ------------------------------------------------------------------
    # INR Yield Curve  (synthetic 3-point from FRED + spread)
    # Assumption: INR yield = USD FRED rate + INR-USD credit spread
    # Documented in docs/model_dev_docs/assumptions.md
    # ------------------------------------------------------------------

    def _ingest_yield_curve(self, start: str, end: str) -> None:
        logger.info("Building synthetic INR yield curve …")
        cfg_yc = self.cfg["yield_curve"]
        tenors = cfg_yc["tenors_years"]
        spreads_bps = cfg_yc["inr_usd_spread_bps"]
        fred_series = cfg_yc["fred_series"]

        rows_all: list[tuple] = []

        try:
            import fredapi  # optional dependency
            fred_client = fredapi.Fred(api_key=self._fred_key) if self._fred_key else None
        except ImportError:
            fred_client = None
            logger.warning("fredapi not installed; using flat USD rate assumption")

        for tenor, spread_bps in zip(tenors, spreads_bps):
            tenor_key = f"{tenor}Y"
            series_id = fred_series.get(tenor_key)
            usd_rates: pd.Series | None = None

            if fred_client and series_id:
                try:
                    usd_rates = fred_client.get_series(
                        series_id, observation_start=start, observation_end=end
                    )
                    usd_rates = usd_rates.dropna() / 100.0   # percent → decimal
                    logger.info("  FRED %s (%s): %d obs", series_id, tenor_key, len(usd_rates))
                except Exception as exc:
                    logger.warning("FRED %s failed: %s", series_id, exc)

            if usd_rates is None or usd_rates.empty:
                # Flat fallback
                dates = pd.bdate_range(start, end)
                usd_rates = pd.Series(0.04, index=dates)
                logger.warning("  Using flat 4%% USD rate for %s", tenor_key)

            for dt, usd_r in usd_rates.items():
                inr_r = float(usd_r) + spread_bps / 10000.0
                rows_all.append(
                    (pd.Timestamp(dt).strftime("%Y-%m-%d"), float(tenor), inr_r, "synthetic_3pt")
                )

        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO yield_curve (date, tenor_years, rate, source) VALUES (?,?,?,?)",
            rows_all,
        )
        self.conn.commit()
        logger.info("  Yield curve: %d rows", len(rows_all))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_hash(self, name: str, df: pd.DataFrame, n_rows: int) -> None:
        h = _sha256_df(df)
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO dataset_hashes (snapshot_name, hash_sha256, n_rows)
               VALUES (?,?,?)""",
            (name, h, n_rows),
        )
        self.conn.commit()

    def load_prices(
        self, tickers: list[str] | None = None, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Load adjusted closes from DB as a wide DataFrame (dates × tickers)."""
        query = "SELECT date, ticker, adj_close FROM prices WHERE 1=1"
        params: list = []
        if tickers:
            placeholders = ",".join("?" * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY date, ticker"

        df = pd.read_sql_query(query, self.conn, params=params)
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.pivot(index="date", columns="ticker", values="adj_close")

    def load_fx(
        self, ccy_pair: str = "USDINR", start: str | None = None, end: str | None = None
    ) -> pd.Series:
        query = "SELECT date, rate FROM fx_rates WHERE ccy_pair=?"
        params: list = [ccy_pair]
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY date"
        df = pd.read_sql_query(query, self.conn, params=params)
        if df.empty:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["rate"]

    def load_vix(
        self, start: str | None = None, end: str | None = None
    ) -> pd.Series:
        query = "SELECT date, value FROM vol_index WHERE index_name='INDIAVIX'"
        params: list = []
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY date"
        df = pd.read_sql_query(query, self.conn, params=params)
        if df.empty:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["value"]

    def load_yield_curve(
        self, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Returns wide DataFrame: index=date, columns=tenor_years."""
        query = "SELECT date, tenor_years, rate FROM yield_curve WHERE 1=1"
        params: list = []
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY date, tenor_years"
        df = pd.read_sql_query(query, self.conn, params=params)
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.pivot(index="date", columns="tenor_years", values="rate")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _safe_int(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None
