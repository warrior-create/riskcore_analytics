"""
Database schema definition and initialization for MarketRisk-Lab.

Tables
------
prices          : daily OHLCV + adjusted close for equities and indices
fx_rates        : daily FX spot rates
yield_curve     : daily INR yield curve (3-point synthetic)
vol_index       : India VIX
positions       : portfolio positions snapshot
pnl_daily       : daily P&L (clean + dirty)
var_results     : daily VaR and ES by model
backtest_results: walk-forward backtest summary
dq_log          : data quality failure log
model_inventory : version-controlled model registry
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- -----------------------------------------------------------------------
-- Core market data
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    adj_close   REAL    NOT NULL,
    volume      INTEGER,
    source      TEXT    DEFAULT 'yfinance',
    loaded_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(date, ticker)
);

CREATE INDEX IF NOT EXISTS ix_prices_date ON prices(date);
CREATE INDEX IF NOT EXISTS ix_prices_ticker ON prices(ticker);

CREATE TABLE IF NOT EXISTS fx_rates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    ccy_pair    TEXT    NOT NULL,        -- e.g. USDINR
    rate        REAL    NOT NULL,
    source      TEXT    DEFAULT 'yfinance',
    loaded_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(date, ccy_pair)
);

CREATE INDEX IF NOT EXISTS ix_fx_date ON fx_rates(date);

CREATE TABLE IF NOT EXISTS yield_curve (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    tenor_years REAL    NOT NULL,
    rate        REAL    NOT NULL,        -- annualised decimal
    source      TEXT    DEFAULT 'synthetic_3pt',
    loaded_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(date, tenor_years)
);

CREATE INDEX IF NOT EXISTS ix_yc_date ON yield_curve(date);

CREATE TABLE IF NOT EXISTS vol_index (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    index_name  TEXT    NOT NULL,        -- INDIAVIX
    value       REAL    NOT NULL,        -- annualised %
    source      TEXT    DEFAULT 'yfinance',
    loaded_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(date, index_name)
);

CREATE INDEX IF NOT EXISTS ix_vol_date ON vol_index(date);

-- -----------------------------------------------------------------------
-- Portfolio
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS positions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date  TEXT    NOT NULL,
    instrument  TEXT    NOT NULL,
    instrument_type TEXT NOT NULL,      -- equity | option | fx_fwd | bond | irs
    quantity    REAL    NOT NULL,
    price       REAL,
    market_value REAL,
    delta       REAL,
    gamma       REAL,
    vega        REAL,
    duration    REAL,
    dv01        REAL,
    loaded_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(as_of_date, instrument)
);

-- -----------------------------------------------------------------------
-- P&L
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pnl_daily (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL UNIQUE,
    clean_pnl       REAL    NOT NULL,   -- hypothetical / full reval
    dirty_pnl       REAL,               -- actual (fees, carry, intraday)
    delta_pnl       REAL,
    gamma_pnl       REAL,
    vega_pnl        REAL,
    rates_pnl       REAL,
    fx_pnl          REAL,
    unexplained_pnl REAL,
    loaded_at       TEXT    DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------
-- VaR / ES results
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS var_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    model           TEXT    NOT NULL,   -- parametric | hs | fhs | mc
    confidence      REAL    NOT NULL,
    holding_period  INTEGER NOT NULL DEFAULT 1,
    var_1d          REAL    NOT NULL,
    es_1d           REAL,
    stressed_es     REAL,
    loaded_at       TEXT    DEFAULT (datetime('now')),
    UNIQUE(date, model, confidence, holding_period)
);

CREATE INDEX IF NOT EXISTS ix_var_date ON var_results(date);

-- -----------------------------------------------------------------------
-- Backtesting
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    test_start      TEXT    NOT NULL,
    test_end        TEXT    NOT NULL,
    n_obs           INTEGER NOT NULL,
    n_exceptions    INTEGER NOT NULL,
    exception_rate  REAL    NOT NULL,
    traffic_light   TEXT    NOT NULL,   -- green | amber | red
    kupiec_stat     REAL,
    kupiec_pvalue   REAL,
    chr_indep_stat  REAL,
    chr_indep_pvalue REAL,
    chr_cc_stat     REAL,
    chr_cc_pvalue   REAL,
    acerbi_z1       REAL,
    acerbi_z2       REAL,
    acerbi_z1_pvalue REAL,
    acerbi_z2_pvalue REAL,
    loaded_at       TEXT    DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------
-- Exception detail
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exception_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    var_estimate    REAL    NOT NULL,
    actual_loss     REAL    NOT NULL,
    excess_loss     REAL    NOT NULL,
    classification  TEXT,               -- vol_regime | single_move | model_lag | position_driver
    narrative       TEXT,
    loaded_at       TEXT    DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------
-- Data Quality
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dq_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at       TEXT    DEFAULT (datetime('now')),
    check_name      TEXT    NOT NULL,
    ticker          TEXT,
    date_affected   TEXT,
    severity        TEXT    NOT NULL,   -- WARNING | ERROR | CRITICAL
    message         TEXT    NOT NULL,
    resolved        INTEGER DEFAULT 0
);

-- -----------------------------------------------------------------------
-- Model Inventory
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_inventory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        TEXT    NOT NULL UNIQUE,  -- e.g. VAR-HS-v2.0
    model_name      TEXT    NOT NULL,
    version         TEXT    NOT NULL,
    tier            TEXT    NOT NULL,         -- 1-High | 2-Medium | 3-Low
    status          TEXT    NOT NULL,         -- development | validation | production | retired
    owner           TEXT,
    validator       TEXT,
    approved_date   TEXT,
    next_review     TEXT,
    description     TEXT,
    limitations     TEXT,
    registered_at   TEXT    DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------
-- Monitoring KRIs
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kri_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    kri_name        TEXT    NOT NULL,
    value           REAL    NOT NULL,
    threshold_amber REAL,
    threshold_red   REAL,
    status          TEXT    NOT NULL,   -- green | amber | red
    alert_generated INTEGER DEFAULT 0,
    logged_at       TEXT    DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------
-- Dataset hashes (snapshot integrity)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dataset_hashes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_name   TEXT    NOT NULL,
    hash_sha256     TEXT    NOT NULL,
    n_rows          INTEGER,
    created_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(snapshot_name)
);

-- -----------------------------------------------------------------------
-- Reporting views
-- -----------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_latest_var AS
SELECT v.*
FROM var_results v
INNER JOIN (
    SELECT model, confidence, MAX(date) as max_date
    FROM var_results
    GROUP BY model, confidence
) latest ON v.model = latest.model
         AND v.confidence = latest.confidence
         AND v.date = latest.max_date;

CREATE VIEW IF NOT EXISTS v_backtest_summary AS
SELECT model,
       confidence,
       COUNT(*) as n_runs,
       AVG(exception_rate) as avg_exception_rate,
       SUM(CASE WHEN traffic_light = 'green' THEN 1 ELSE 0 END) as green_windows,
       SUM(CASE WHEN traffic_light = 'amber' THEN 1 ELSE 0 END) as amber_windows,
       SUM(CASE WHEN traffic_light = 'red' THEN 1 ELSE 0 END) as red_windows
FROM backtest_results
GROUP BY model, confidence;

CREATE VIEW IF NOT EXISTS v_dq_open_issues AS
SELECT * FROM dq_log WHERE resolved = 0 ORDER BY logged_at DESC;
"""


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialise the SQLite database and return a connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Return a connection to an existing database."""
    db_path = Path(db_path)
    if not db_path.exists():
        return init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn
