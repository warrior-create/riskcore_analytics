"""
Excel report generator using openpyxl.

Generates a multi-sheet Excel workbook:
  Sheet 1: Portfolio Summary
  Sheet 2: VaR / ES Results (all models)
  Sheet 3: Backtesting Summary (traffic light, Kupiec, Christoffersen)
  Sheet 4: Exception Log
  Sheet 5: KRI Monitoring
  Sheet 6: Stressed ES

Styling: conditional formatting (red/amber/green), charts.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import (
    Alignment, Font, PatternFill, Border, Side, numbers
)
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule

logger = logging.getLogger(__name__)

# Colour palette
COLORS = {
    "green_fill":  PatternFill("solid", fgColor="C6EFCE"),
    "amber_fill":  PatternFill("solid", fgColor="FFEB9C"),
    "red_fill":    PatternFill("solid", fgColor="FFC7CE"),
    "header_fill": PatternFill("solid", fgColor="1F3864"),
    "alt_fill":    PatternFill("solid", fgColor="EBF0F5"),
}

HEADER_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
BODY_FONT = Font(name="Calibri", size=10)
TITLE_FONT = Font(name="Calibri", bold=True, size=14, color="1F3864")

THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)


class ExcelReporter:
    """Generate the daily Excel risk report."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = Path(cfg.get("reporting", {}).get("excel_output_dir", "data/reports"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, report_date: str) -> str:
        """Generate report for a given date. Returns output file path."""
        wb = openpyxl.Workbook()
        wb.remove(wb.active)   # remove default empty sheet

        # Load data from DB
        from marketrisk.data.schema import get_connection
        conn = get_connection(self.cfg["data"]["db_path"])

        self._sheet_summary(wb, conn, report_date)
        self._sheet_var_results(wb, conn, report_date)
        self._sheet_backtest(wb, conn)
        self._sheet_exceptions(wb, conn)
        self._sheet_kri(wb, conn)
        self._sheet_metadata(wb, report_date)

        conn.close()

        fname = f"risk_report_{report_date.replace('-', '')}.xlsx"
        out_path = self.out_dir / fname
        wb.save(str(out_path))
        logger.info("Excel report saved: %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Sheet builders
    # ------------------------------------------------------------------

    def _sheet_summary(self, wb, conn, report_date: str) -> None:
        ws = wb.create_sheet("Portfolio Summary")
        ws.sheet_view.showGridLines = False

        ws["B2"] = "MarketRisk-Lab | Daily Risk Report"
        ws["B2"].font = TITLE_FONT
        ws["B3"] = f"Report Date: {report_date}"
        ws["B3"].font = BODY_FONT

        # VaR summary table
        try:
            df = pd.read_sql_query(
                "SELECT * FROM v_latest_var ORDER BY model, confidence",
                conn
            )
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            headers = ["Model", "Confidence", "VaR (1-day)", "ES (1-day)", "Date"]
            cols = ["model", "confidence", "var_1d", "es_1d", "date"]
            self._write_table(ws, df[cols], headers, start_row=6, start_col=2)

        ws.column_dimensions["A"].width = 3
        ws.column_dimensions["B"].width = 20
        ws.column_dimensions["C"].width = 15
        ws.column_dimensions["D"].width = 15
        ws.column_dimensions["E"].width = 15
        ws.column_dimensions["F"].width = 20

    def _sheet_var_results(self, wb, conn, report_date: str) -> None:
        ws = wb.create_sheet("VaR Results")
        ws.sheet_view.showGridLines = False
        try:
            df = pd.read_sql_query(
                """SELECT date, model, confidence, var_1d, es_1d
                   FROM var_results
                   WHERE date <= ?
                   ORDER BY date DESC, model, confidence
                   LIMIT 500""",
                conn,
                params=(report_date,),
            )
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            self._write_table(ws, df, list(df.columns), start_row=2, start_col=2)

    def _sheet_backtest(self, wb, conn) -> None:
        ws = wb.create_sheet("Backtesting")
        ws.sheet_view.showGridLines = False
        try:
            df = pd.read_sql_query(
                "SELECT * FROM v_backtest_summary", conn
            )
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            self._write_table(ws, df, list(df.columns), start_row=2, start_col=2)

    def _sheet_exceptions(self, wb, conn) -> None:
        ws = wb.create_sheet("Exceptions")
        ws.sheet_view.showGridLines = False
        try:
            df = pd.read_sql_query(
                """SELECT date, model, confidence, var_estimate, actual_loss,
                          excess_loss, classification, narrative
                   FROM exception_log
                   ORDER BY date DESC
                   LIMIT 200""",
                conn,
            )
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            self._write_table(ws, df, list(df.columns), start_row=2, start_col=2)
            # Conditional format classification column
            col_letter = get_column_letter(2 + 6)   # classification col
            for row in range(3, 3 + len(df)):
                cell = ws[f"{col_letter}{row}"]
                if cell.value == "vol_regime":
                    cell.fill = COLORS["red_fill"]
                elif cell.value == "model_lag":
                    cell.fill = COLORS["amber_fill"]

    def _sheet_kri(self, wb, conn) -> None:
        ws = wb.create_sheet("KRI Monitoring")
        ws.sheet_view.showGridLines = False
        try:
            df = pd.read_sql_query(
                "SELECT * FROM kri_log ORDER BY date DESC LIMIT 300", conn
            )
        except Exception:
            df = pd.DataFrame()

        if not df.empty:
            self._write_table(ws, df, list(df.columns), start_row=2, start_col=2)

    def _sheet_metadata(self, wb, report_date: str) -> None:
        ws = wb.create_sheet("Metadata")
        ws["B2"] = "MarketRisk-Lab Report Metadata"
        ws["B2"].font = TITLE_FONT
        meta = {
            "Report Date": report_date,
            "Generated At": datetime.utcnow().isoformat() + "Z",
            "Version": "0.1.0",
            "Random Seed": 42,
            "Yield Curve Source": "Synthetic 3-point (FRED USD + INR spread)",
            "Option Pricing": "Black-76 with linear skew",
            "Compliance": "Simplified FRTB (documented assumptions)",
        }
        for i, (k, v) in enumerate(meta.items(), start=4):
            ws[f"B{i}"] = k
            ws[f"C{i}"] = str(v)
            ws[f"B{i}"].font = Font(bold=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_table(
        self,
        ws,
        df: pd.DataFrame,
        headers: list[str],
        start_row: int = 1,
        start_col: int = 1,
    ) -> None:
        # Header row
        for j, h in enumerate(headers):
            cell = ws.cell(row=start_row, column=start_col + j, value=h)
            cell.font = HEADER_FONT
            cell.fill = COLORS["header_fill"]
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = THIN_BORDER

        # Data rows
        for i, (_, row) in enumerate(df.iterrows()):
            fill = COLORS["alt_fill"] if i % 2 == 0 else None
            for j, val in enumerate(row.values):
                cell = ws.cell(row=start_row + 1 + i, column=start_col + j)
                if isinstance(val, float):
                    cell.value = round(val, 6)
                    cell.number_format = "0.0000"
                else:
                    cell.value = val
                cell.font = BODY_FONT
                cell.border = THIN_BORDER
                if fill:
                    cell.fill = fill
                cell.alignment = Alignment(vertical="center")
