"""
MarketRisk-Lab CLI.

Usage
-----
    marketrisk init-db
    marketrisk ingest [--start YYYY-MM-DD] [--end YYYY-MM-DD]
    marketrisk run-daily [--date YYYY-MM-DD] [--config CONFIG]
    marketrisk backtest [--model MODEL] [--window rolling|expanding]
    marketrisk report [--date YYYY-MM-DD] [--format excel|pdf]
    marketrisk check-dq
    marketrisk show-findings

All commands read from config/settings.yaml by default.
Fixed random seed (42) applied globally.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Fix random seed globally per spec
np.random.seed(42)

console = Console()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        format="%(message)s",
        datefmt="[%X]",
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", default="config/settings.yaml", help="Path to settings.yaml")
@click.option("--verbose", is_flag=True, default=False)
@click.pass_context
def main(ctx, config, verbose):
    """MarketRisk-Lab: End-to-End Market Risk Framework."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["cfg"] = load_config(config)


# ---------------------------------------------------------------------------
# init-db
# ---------------------------------------------------------------------------

@main.command("init-db")
@click.pass_context
def init_db(ctx):
    """Initialise the SQLite database and schema."""
    cfg = ctx.obj["cfg"]
    from marketrisk.data.schema import init_db as _init_db
    conn = _init_db(cfg["data"]["db_path"])
    console.print(f"[green]✓[/green] Database initialised at [bold]{cfg['data']['db_path']}[/bold]")
    conn.close()


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@main.command("ingest")
@click.option("--start", default=None, help="Start date YYYY-MM-DD")
@click.option("--end", default=None, help="End date YYYY-MM-DD")
@click.pass_context
def ingest(ctx, start, end):
    """Download all market data and store in DB."""
    cfg = ctx.obj["cfg"]
    from marketrisk.data.ingest import DataIngestor
    with console.status("[bold cyan]Downloading data…"):
        ingst = DataIngestor(cfg)
        ingst.run_full(start=start, end=end)
    console.print("[green]✓[/green] Data ingestion complete.")


# ---------------------------------------------------------------------------
# check-dq
# ---------------------------------------------------------------------------

@main.command("check-dq")
@click.pass_context
def check_dq(ctx):
    """Run all data-quality checks and display open issues."""
    cfg = ctx.obj["cfg"]
    from marketrisk.data.ingest import DataIngestor
    from marketrisk.data.quality import DataQualityChecker
    from marketrisk.data.calendar import NSECalendar

    ingst = DataIngestor(cfg)
    tickers = cfg["data"]["equity_tickers"] + [cfg["data"]["nifty_ticker"]]
    prices = ingst.load_prices(tickers)

    if prices.empty:
        console.print("[red]No price data found. Run 'ingest' first.[/red]")
        return

    nifty_prices = ingst.load_prices([cfg["data"]["nifty_ticker"]])
    cal = NSECalendar(nifty_prices)
    checker = DataQualityChecker(ingst.conn, cal, cfg)
    issues = checker.run_all(prices)

    if issues.empty:
        console.print("[green]✓[/green] All DQ checks passed.")
    else:
        table = Table(title=f"DQ Issues ({len(issues)})")
        for col in ["check_name", "ticker", "date_affected", "severity", "message"]:
            table.add_column(col)
        for _, row in issues.head(20).iterrows():
            table.add_row(*[str(row.get(c, "")) for c in ["check_name", "ticker", "date_affected", "severity", "message"]])
        console.print(table)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@main.command("backtest")
@click.option("--model", default=None, help="Model to run (all if omitted)")
@click.option("--window", default="rolling", type=click.Choice(["rolling", "expanding"]))
@click.pass_context
def backtest(ctx, model, window):
    """Run walk-forward backtesting."""
    cfg = ctx.obj["cfg"]
    from marketrisk.data.ingest import DataIngestor
    from marketrisk.engine.walk_forward import WalkForwardEngine
    from marketrisk.backtest.statistical_tests import run_backtest_battery

    ingst = DataIngestor(cfg)
    tickers = cfg["data"]["equity_tickers"] + [cfg["data"]["nifty_ticker"]]
    prices = ingst.load_prices(tickers)

    if prices.empty:
        console.print("[red]No data. Run ingest first.[/red]")
        return

    returns = np.log(prices / prices.shift(1)).dropna()
    models = [model] if model else None

    with console.status(f"[bold cyan]Running walk-forward ({window})…"):
        engine = WalkForwardEngine(
            returns=returns,
            portfolio_weights=None,
            cfg=cfg,
            conn=ingst.conn,
        )
        results = engine.run(window_type=window, models=models)

    # Print summary
    if results.empty:
        console.print("[red]No results produced.[/red]")
        return

    # Synthetic P&L from returns
    pnl = returns.mean(axis=1)

    summary = run_backtest_battery(results, pnl, cfg)
    
    # --- ADDED: Save to Database ---
    try:
        # 1. Backtest Results
        summary["run_date"] = str(pd.Timestamp.today().date())
        summary.to_sql("backtest_results", ingst.conn, if_exists="append", index=False)
        
        # Prepare results dataframe for ExceptionEngine and Monitoring
        results_merged = results.set_index("date").join(pnl.rename("actual_pnl"), how="left").reset_index()
        results_merged["actual_loss"] = -results_merged["actual_pnl"]
        results_merged["exception"] = results_merged["actual_loss"] > results_merged["var_1d"]

        # 2. Exceptions
        from marketrisk.backtest.exceptions import ExceptionAnalysisEngine
        vix_ticker = cfg["data"].get("vix_ticker")
        vix = prices[vix_ticker] if vix_ticker and vix_ticker in prices else None
        ex_engine = ExceptionAnalysisEngine(results_merged, returns, vix)
        ex_engine.run(conn=ingst.conn)
        
        # 3. KRIs
        from marketrisk.backtest.monitoring import MonitoringFramework
        mon = MonitoringFramework(results_merged, pnl, cfg)
        for m in (models or ["hs", "fhs_ewma", "parametric"]):
            kri_df = mon.run(model=m, confidence=0.99)
            if not kri_df.empty:
                # Pivot from wide to long format to match DB schema
                long_kri = []
                for _, row in kri_df.iterrows():
                    for kri_name in ["rolling_breach_rate", "rolling_kupiec_pvalue", "model_stability", "var_pnl_ratio"]:
                        if kri_name in row:
                            long_kri.append({
                                "date": str(row["date"])[:10],
                                "kri_name": kri_name,
                                "value": row[kri_name],
                                "status": row["status"]
                            })
                if long_kri:
                    pd.DataFrame(long_kri).to_sql("kri_log", ingst.conn, if_exists="append", index=False)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save all results to DB: {e}[/yellow]")
    # -------------------------------

    table = Table(title="Backtest Summary")
    for col in summary.columns:
        table.add_column(col, no_wrap=True)
    for _, row in summary.iterrows():
        vals = []
        for c in summary.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        color = {"green": "green", "amber": "yellow", "red": "red"}.get(
            row.get("traffic_light", ""), "white"
        )
        table.add_row(*vals, style=color)

    console.print(table)
    console.print(f"[green]✓[/green] Backtest complete. {len(results)} VaR estimates computed.")


# ---------------------------------------------------------------------------
# run-daily
# ---------------------------------------------------------------------------

@main.command("run-daily")
@click.option("--date", default=None, help="Report date YYYY-MM-DD (default: today)")
@click.pass_context
def run_daily(ctx, date):
    """Run the full daily risk pipeline: ingest → compute → backtest → report."""
    cfg = ctx.obj["cfg"]
    target_date = date or pd.Timestamp.today().strftime("%Y-%m-%d")

    console.print(f"[bold cyan]Daily run for {target_date}[/bold cyan]")

    # 1. Ensure DB exists
    from marketrisk.data.schema import init_db as _init_db
    conn = _init_db(cfg["data"]["db_path"])
    conn.close()

    # 2. Ingest (idempotent)
    ctx.invoke(ingest, start=None, end=target_date)

    # 3. DQ check
    ctx.invoke(check_dq)

    # 4. Backtest
    ctx.invoke(backtest, model=None, window="rolling")

    # 5. Report
    ctx.invoke(report, date=target_date, fmt="excel")

    console.print(f"[bold green]✓ Daily run complete for {target_date}[/bold green]")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@main.command("report")
@click.option("--date", default=None)
@click.option("--format", "fmt", default="excel", type=click.Choice(["excel", "pdf"]))
@click.option("--email", default=None, help="Email address to send the report to")
@click.pass_context
def report(ctx, date, fmt, email):
    """Generate Excel or PDF risk report."""
    cfg = ctx.obj["cfg"]
    target_date = date or pd.Timestamp.today().strftime("%Y-%m-%d")

    if fmt == "excel":
        from marketrisk.reporting.excel_report import ExcelReporter
        reporter = ExcelReporter(cfg)
        path = reporter.generate(target_date)
    else:
        from marketrisk.reporting.pdf_report import PDFReporter
        reporter = PDFReporter(cfg)
        path = reporter.generate(target_date)
        
    console.print(f"[green]✓[/green] Report written to [bold]{path}[/bold]")
    
    if email:
        console.print(f"[cyan]Sending report to {email}...[/cyan]")
        import smtplib
        from email.message import EmailMessage
        import mimetypes
        
        try:
            msg = EmailMessage()
            msg['Subject'] = f"MarketRisk-Lab Daily Report: {target_date}"
            msg['From'] = "risk-system@marketrisk-lab.internal"
            msg['To'] = email
            msg.set_content(f"Please find attached the daily market risk report for {target_date}.")
            
            ctype, encoding = mimetypes.guess_type(path)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            
            with open(path, 'rb') as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path))
            
            # Send using a local or configured SMTP server. 
            # We assume localhost on port 1025 (like MailHog) or standard mock setup for demo.
            smtp_host = cfg.get("smtp_host", "localhost")
            smtp_port = cfg.get("smtp_port", 1025)
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.send_message(msg)
                
            console.print(f"[green]✓[/green] Report successfully emailed to [bold]{email}[/bold]")
        except Exception as e:
            console.print(f"[red]Failed to send email: {e}[/red]")


# ---------------------------------------------------------------------------
# show-findings
# ---------------------------------------------------------------------------

@main.command("show-findings")
def show_findings():
    """Display the FINDINGS.md summary in the terminal."""
    findings_path = Path("FINDINGS.md")
    if findings_path.exists():
        console.print(findings_path.read_text())
    else:
        console.print("[yellow]FINDINGS.md not found. Run backtest first.[/yellow]")


if __name__ == "__main__":
    main()
