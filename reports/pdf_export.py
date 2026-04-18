"""
reports/pdf_export.py
----------------------
Generates a professional PDF legal risk report using fpdf2 (free, no API needed).

The PDF includes:
- Cover page with risk index
- Executive summary
- Per-clause risk cards (for high-risk clauses)
- Recommendations
- Legal disclaimer footer
"""

import logging
import io
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_pdf_report(report: Dict[str, Any]) -> bytes:
    """
    Generates a complete PDF legal risk report from the structured report dict.

    Args:
        report: The output of reports.json_report.build_report()

    Returns:
        PDF file as bytes (ready for st.download_button)
    """
    try:
        from fpdf import FPDF
        return _build_pdf(report, FPDF)
    except ImportError:
        logger.error("[PDF] fpdf2 not installed. Returning fallback text report.")
        return _text_fallback(report)
    except Exception as e:
        logger.error(f"[PDF] Generation failed: {e}")
        return _text_fallback(report)


# ══════════════════════════════════════════════════════════════════
# PDF Builder
# ══════════════════════════════════════════════════════════════════

def _build_pdf(report: Dict[str, Any], FPDF) -> bytes:
    """Core PDF generation logic using fpdf2."""

    meta  = report.get("report_metadata", {})
    stats = report.get("statistics", {})
    risks = report.get("identified_risks", [])

    # ── Colour palette ─────────────────────────────────────────────────
    C_BG     = (13,  17,  23)    # Dark navy background
    C_NAVY   = (22,  27,  34)    # Card background
    C_BLUE   = (79,  142, 247)   # Accent blue
    C_RED    = (239, 68,  68)    # High risk red
    C_GREEN  = (34,  197, 94)    # Low risk green
    C_AMBER  = (245, 158, 11)    # Moderate amber
    C_TEXT   = (230, 237, 243)   # Body text
    C_MUTED  = (139, 148, 158)   # Muted labels

    class LegalPDF(FPDF):
        def header(self):
            # Coloured top bar
            self.set_fill_color(*C_NAVY)
            self.rect(0, 0, 210, 12, 'F')
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*C_MUTED)
            self.set_xy(10, 3)
            self.cell(0, 6, "LEXIQ SAAS PLATFORM  |  Milestone 3.0", ln=0)
            self.set_xy(0, 3)
            self.set_font("Helvetica", "", 7)
            self.cell(200, 6, meta.get("generated_at", ""), align="R", ln=0)

        def footer(self):
            self.set_y(-15)
            self.set_fill_color(*C_NAVY)
            self.rect(0, self.get_y(), 210, 15, 'F')
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(*C_MUTED)
            self.cell(0, 10, "⚠  AI-Generated. Not Legal Advice. Consult a licensed attorney.", align="C", ln=0)

    pdf = LegalPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Page background ────────────────────────────────────────────────
    pdf.set_fill_color(*C_BG)
    pdf.rect(0, 12, 210, 285, 'F')

    # ── Cover Section ──────────────────────────────────────────────────
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*C_BLUE)
    pdf.cell(0, 12, "Legal Risk Analysis Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 7, meta.get("document_name", "Unknown Document"), ln=True, align="C")
    pdf.ln(4)

    # Risk Index badge
    risk_idx = stats.get("risk_index", 0)
    idx_color = C_RED if risk_idx >= 7 else (C_AMBER if risk_idx >= 4 else C_GREEN)
    pdf.set_fill_color(*idx_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    x_center = (210 - 60) / 2
    pdf.set_xy(x_center, pdf.get_y())
    pdf.cell(60, 12, f"Risk Index: {risk_idx}/10", align="C", fill=True, ln=True)
    pdf.ln(6)

    # ── Statistics Row ─────────────────────────────────────────────────
    _stat_card(pdf, "Total Clauses",    str(stats.get("total_clauses", 0)),     10, pdf.get_y(), C_NAVY, C_TEXT)
    _stat_card(pdf, "High Risk",        str(stats.get("high_risk_clauses", 0)), 75, pdf.get_y()-20, C_NAVY, C_RED)
    _stat_card(pdf, "Low Risk",         str(stats.get("low_risk_clauses", 0)),  140, pdf.get_y()-20, C_NAVY, C_GREEN)
    pdf.ln(26)

    # ── Executive Summary ──────────────────────────────────────────────
    pdf.set_y(pdf.get_y())
    _section_header(pdf, "Executive Summary", C_BLUE, C_BG)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*C_TEXT)
    pdf.multi_cell(0, 5.5, report.get("contract_summary", ""), align="J")
    pdf.ln(5)

    # ── Risk Findings ──────────────────────────────────────────────────
    high_risks = [r for r in risks if r.get("risk_level") == "High Risk"]
    if high_risks:
        _section_header(pdf, f"High-Risk Clauses ({len(high_risks)} found)", C_RED, C_BG)
        for item in high_risks[:10]:  # max 10 in PDF
            _risk_card(pdf, item, C_NAVY, C_RED, C_TEXT, C_MUTED, C_BLUE)

    # ── Recommendations ────────────────────────────────────────────────
    _section_header(pdf, "Recommendations", C_BLUE, C_BG)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*C_TEXT)
    pdf.multi_cell(0, 5.5, report.get("recommendations", ""), align="J")

    # ── Disclaimer ────────────────────────────────────────────────────
    pdf.ln(5)
    _section_header(pdf, "Disclaimer", C_AMBER, C_BG)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(*C_MUTED)
    pdf.multi_cell(0, 5, report.get("disclaimer", ""), align="J")

    # Return as bytes
    return bytes(pdf.output())


def _section_header(pdf, title: str, color: tuple, bg: tuple):
    """Renders a coloured section header bar."""
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, f"  {title}", fill=True, ln=True)
    pdf.set_fill_color(*bg)
    pdf.ln(2)


def _stat_card(pdf, label: str, value: str, x: float, y: float, bg: tuple, val_color: tuple):
    """Renders a small statistics card at the given position."""
    pdf.set_xy(x, y)
    pdf.set_fill_color(*bg)
    pdf.set_draw_color(48, 54, 61)
    pdf.rect(x, y, 55, 18, 'FD')
    # Label
    pdf.set_xy(x + 2, y + 2)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(139, 148, 158)
    pdf.cell(51, 4, label.upper(), ln=True)
    # Value
    pdf.set_xy(x + 2, y + 8)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*val_color)
    pdf.cell(51, 8, value, ln=False)


def _risk_card(pdf, item: dict, bg, red, text, muted, blue):
    """Renders a single clause risk card."""
    if pdf.get_y() > 250:
        pdf.add_page()
        # Reapply background
        pdf.set_fill_color(*bg)
        pdf.rect(0, 12, 210, 285, 'F')

    y_start = pdf.get_y()
    pdf.set_fill_color(*bg)
    pdf.set_draw_color(*red)
    pdf.round_corner_rect = None  # Use standard rect

    # Card border
    card_h = 52
    pdf.set_xy(8, y_start)
    pdf.set_fill_color(22, 27, 34)
    pdf.rect(8, y_start, 194, card_h, 'F')
    pdf.set_draw_color(*red)
    pdf.rect(8, y_start, 3, card_h, 'F')  # Left accent bar

    # Clause number + risk badge
    pdf.set_xy(14, y_start + 2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*red)
    pdf.cell(40, 5, f"Clause #{item.get('clause_number', '?')}  ●  {item.get('risk_level', '')}", ln=False)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*muted)
    pdf.set_x(150)
    pdf.cell(50, 5, f"Confidence: {item.get('confidence', 'N/A')}", ln=True)

    # Clause text snippet
    clause_snippet = item.get("clause", "")[:180].replace("\n", " ")
    pdf.set_xy(14, y_start + 8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*muted)
    pdf.multi_cell(188, 4.5, f'"{clause_snippet}..."', align="L")

    # Explanation
    explanation = item.get("explanation", "")[:200]
    pdf.set_xy(14, y_start + 24)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*blue)
    pdf.cell(0, 4, "Analysis:", ln=True)
    pdf.set_xy(14, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*text)
    pdf.multi_cell(188, 4.5, explanation, align="L")

    # Mitigation
    mitigation = item.get("mitigation", "")[:160]
    pdf.set_xy(14, pdf.get_y() + 1)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(34, 197, 94)
    pdf.cell(0, 4, "Mitigation:", ln=True)
    pdf.set_xy(14, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*text)
    pdf.multi_cell(188, 4.5, mitigation, align="L")

    pdf.ln(4)


# ══════════════════════════════════════════════════════════════════
# Text Fallback (if fpdf2 unavailable)
# ══════════════════════════════════════════════════════════════════

def _text_fallback(report: Dict[str, Any]) -> bytes:
    """Returns a plain-text version of the report as bytes."""
    lines = [
        "INTELLIGENT CONTRACT RISK ANALYSIS REPORT",
        "=" * 60,
        f"Document: {report.get('report_metadata', {}).get('document_name', 'N/A')}",
        f"Generated: {report.get('report_metadata', {}).get('generated_at', 'N/A')}",
        "",
        "EXECUTIVE SUMMARY",
        report.get("contract_summary", ""),
        "",
        "HIGH-RISK CLAUSES",
    ]
    for risk in report.get("identified_risks", []):
        if risk.get("risk_level") == "High Risk":
            lines += [
                f"\nClause #{risk['clause_number']} — {risk['risk_level']} ({risk['confidence']})",
                f"Clause: {risk['clause'][:200]}...",
                f"Analysis: {risk.get('explanation', '')}",
                f"Mitigation: {risk.get('mitigation', '')}",
            ]
    lines += [
        "",
        "RECOMMENDATIONS",
        report.get("recommendations", ""),
        "",
        report.get("disclaimer", ""),
    ]
    return "\n".join(lines).encode("utf-8")
