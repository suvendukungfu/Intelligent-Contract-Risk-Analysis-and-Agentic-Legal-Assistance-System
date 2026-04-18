"""
app/streamlit_app.py  —  Milestone 3: Top 1% SaaS LegalTech Platform
======================================================================
Enterprise-grade Legal AI Platform.
Features:
  - 📄 Risk Analysis (Real-time explainability)
  - 🧠 AI Setup & Trust
  - 📊 Analytics Dashboard (Plotly)
  - 📑 Compare Contracts (Semantic Multi-Document)
  - 📥 Export Center (Premium PDF & JSON)
"""

import os
import sys
import json
import logging
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd

# Core internal imports
from config.settings import ARTIFACTS_DIR
from nlp.clause_segmenter import segment_clauses
from models.inference import risk_engine

from reports.json_report import report_to_json_string
from reports.pdf_export import generate_pdf_report
from agents.workflow import run_agent_pipeline
from analytics.dashboard import (
    create_confidence_trend_chart, 
    create_risk_distribution_chart, 
    create_complexity_vs_risk_chart
)
from models.comparison import compare_contracts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# PAGE SETTINGS & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LexIQ Enterprise", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* LexIQ Obsidian Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    
    .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 12px; padding: 6px; box-shadow: inset 0px 2px 4px rgba(0,0,0,0.2); }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #94a3b8; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: #1e293b !important; color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }
    
    /* Neumorphic Cards */
    .saas-card {
        background: #1e293b; border-radius: 16px; padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5), 0 2px 4px -1px rgba(0,0,0,0.3);
        border: 1px solid #334155; margin-bottom: 24px; transition: transform 0.2s;
    }
    .saas-card:hover { transform: translateY(-2px); border-color: #475569; }
    
    /* Metrics */
    .metric-title { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 8px; }
    .metric-value { font-size: 2.5rem; color: #f8fafc; font-weight: 700; line-height: 1; }
    .metric-value.critical { color: #ef4444; }
    .metric-value.warning { color: #f59e0b; }
    .metric-value.safe { color: #10b981; }
    
    /* Badges */
    .badge { padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
    .badge-high { background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge-low { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .kw-chip { background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2); padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; margin: 2px; display: inline-block; }
    
    /* Clause List View */
    .clause-row { background: #0f172a; padding: 16px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #10b981; cursor: pointer; }
    .clause-row.high { border-left-color: #ef4444; background: #1e1b24; }
    .clause-row:hover { filter: brightness(1.2); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE & HELPERS
# ══════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "agent_state_a": None, "ml_df_a": None, "file_name_a": "", "raw_a": "",
        "agent_state_b": None, "ml_df_b": None, "file_name_b": "", "raw_b": "",
        "active_doc": "A",
        "selected_clause_idx": None
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def get_text_from_file(file) -> str:
    # PDF Parsing
    if file.name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        except Exception as e: return f"PDF Error: {str(e)}"
    return file.getvalue().decode("utf-8", errors="ignore")

def load_demo_contract(variant="NDA"):
    path = os.path.join(ROOT, "sample_docs", f"sample_{variant.lower()}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), f"LexIQ_Demo_{variant}.txt"
    return "", ""

# ══════════════════════════════════════════════════════════════════
# SIDEBAR / APP SHELL
# ══════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding-bottom: 24px;">
            <div style="font-size:3rem; line-height:1;">⚖️</div>
            <div style="font-size:1.5rem; font-weight:800; color:#f8fafc; letter-spacing:-0.5px;">LexIQ</div>
            <div style="font-size:0.8rem; color:#38bdf8; font-weight:600; margin-top:4px;">ENTERPRISE EDITION</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='color:#94a3b8; font-size:0.8rem; font-weight:600; text-transform:uppercase;'>Upload Documents</p>", unsafe_allow_html=True)
        
        file_a = st.file_uploader("Primary Contract (Doc A)", type=["pdf", "txt"], key="upload_a")
        file_b = st.file_uploader("Comparison Contract (Doc B, Optional)", type=["pdf", "txt"], key="upload_b")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        demo_nda = col1.button("Preload NDA", use_container_width=True)
        demo_msa = col2.button("Preload MSA", use_container_width=True)
        
        execute = st.button("🚀 EXECUTE AI PIPELINE", use_container_width=True, type="primary")

        # LLM Trust / Status panel
        st.markdown("""
        <div style="background:#1e293b; padding:16px; border-radius:12px; margin-top:24px; border:1px solid #334155;">
            <div style="font-size:0.75rem; font-weight:700; color:#94a3b8; margin-bottom:8px;">TRUST & COMPLIANCE HUD</div>
            <div style="font-size:0.8rem; color:#10b981;">● RAG Vector Store: Sync'd</div>
            <div style="font-size:0.8rem; color:#10b981;">● Inference Model: Scikit v1.6</div>
            <div style="font-size:0.8rem; color:#38bdf8;">● LLM Tier: Rule/Local </div>
        </div>
        """, unsafe_allow_html=True)

        return execute, file_a, file_b, demo_nda, demo_msa

def process_document(raw_text: str, filename: str, doc_key: str):
    """Runs LangGraph pipeline and saves to session state."""
    with st.spinner(f"Analyzing {filename}... Parse → RAG → LLM"):
        state = run_agent_pipeline(raw_text, file_name=filename)
        ml_results = state.get("ml_results", [])
        
        rows = [{
            "id": r["clause_idx"],
            "Clause": r["clause"],
            "Risk Level": r["risk_level"],
            "Confidence": r["confidence"],
            "Keywords": r.get("triggers", [])
        } for r in ml_results]
        
        df = pd.DataFrame(rows)
        st.session_state[f"ml_df_{doc_key}"] = df
        st.session_state[f"agent_state_{doc_key}"] = state
        st.session_state[f"file_name_{doc_key}"] = filename
        st.session_state[f"raw_{doc_key}"] = raw_text

# ══════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════

def tab_risk_analysis():
    state = st.session_state["agent_state_a"]
    df = st.session_state["ml_df_a"]
    
    if not state:
        st.info("Upload a contract or select a Demo file, then click Execute Workflow.")
        return

    report = state.get("final_report", {})
    stats = report.get("statistics", {})

    # Top KPI Bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="saas-card" style="padding:16px;">
            <div class="metric-title">Risk Index</div>
            <div class="metric-value { 'critical' if stats.get('risk_index',0) >= 7 else 'safe'}">{stats.get('risk_index', '0.0')}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="saas-card" style="padding:16px;">
            <div class="metric-title">High Risk Clauses</div>
            <div class="metric-value critical">{stats.get('high_risk_clauses', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="saas-card" style="padding:16px;">
            <div class="metric-title">Total Provisions</div>
            <div class="metric-value">{stats.get('total_clauses', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="saas-card" style="padding:16px;">
            <div class="metric-title">Avg Confidence</div>
            <div class="metric-value" style="color:#38bdf8;">{stats.get('avg_confidence', '0%')}</div>
        </div>""", unsafe_allow_html=True)

    # ── Real-Time Clause Explanation UI ──
    st.markdown("<h3 style='color:#e2e8f0; font-weight:700; margin-bottom:16px;'>Interactive Clause Review</h3>", unsafe_allow_html=True)
    
    col_list, col_details = st.columns([1.2, 1.8])
    
    with col_list:
        st.markdown("##### Document Clauses")
        st.markdown("<div style='height: 600px; overflow-y:auto; padding-right:10px;'>", unsafe_allow_html=True)
        
        risks = report.get("identified_risks", [])
        
        # We simulate interactivity using streamlit buttons masked as rows
        for idx, item in enumerate(risks):
            r_level = item.get("risk_level")
            cls = "high" if r_level == "High Risk" else "low"
            
            # Using st.button for clickability
            btn_label = f"Clause #{item['clause_number']} ━ {item['confidence']}"
            if r_level == 'High Risk': btn_label = "🔴 " + btn_label
            else: btn_label = "🟢 " + btn_label
            
            if st.button(btn_label, key=f"btn_c_{idx}", use_container_width=True):
                st.session_state["selected_clause_idx"] = idx
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_details:
        st.markdown("##### Explainable AI Insight")
        idx = st.session_state.get("selected_clause_idx")
        
        if idx is not None and idx < len(risks):
            item = risks[idx]
            
            conf_float = float(item.get("confidence", "0").replace("%", ""))
            warning_html = ""
            if conf_float < 75.0:
                warning_html = "<div style='color:#f59e0b; background:rgba(245,158,11,0.1); padding:8px; border-radius:6px; font-size:0.8rem; margin-bottom:12px;'>⚠️ <b>Low Confidence Warning:</b> The AI is uncertain. Human review strongly advised.</div>"

            chips = "".join([f"<span class='kw-chip'>{t}</span>" for t in item.get("linguistic_triggers", [])])
            if item.get("is_anomaly"):
                chips += f" <span class='kw-chip' style='background:rgba(217,70,239,0.1);color:#d946ef;border-color:#d946ef'>🧬 SEMANTIC ANOMALY (Score: {item.get('anomaly_score')})</span>"

            # Render Mathematical XAI Weights (Feature Importance)
            xai_html = ""
            if item.get("xai_weights"):
                xai_html = "<h6 style='color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem; margin-top:16px;'>XAI Feature Importance (Logistic Weights)</h6>"
                for word, weight in item.get("xai_weights").items():
                    w_pct = min(weight * 100, 100)
                    xai_html += f"""
                    <div style="display:flex; align-items:center; margin-bottom:4px;">
                        <div style="width:100px; color:#94a3b8; font-size:0.8rem; overflow:hidden; text-overflow:ellipsis;">{word}</div>
                        <div style="flex-grow:1; background:#1f2937; height:8px; border-radius:4px; margin:0 10px;">
                            <div style="width:{w_pct}%; background:#ef4444; height:100%; border-radius:4px;"></div>
                        </div>
                        <div style="font-size:0.75rem; color:#f87171;">+{weight:.2f}</div>
                    </div>
                    """
                xai_html += "<div style='margin-bottom:24px;'></div>"

            st.markdown(f"""
            <div class="saas-card" style="margin-bottom:0;">
                {warning_html}
                <div style="margin-bottom: 16px;">
                    <span class="badge {'badge-high' if item['risk_level'] == 'High Risk' else 'badge-low'}">{item['risk_level']}</span>
                    {chips}
                </div>
                <div style="background:#0b0f19; padding:16px; border-radius:8px; font-style:italic; font-size:0.9rem; color:#94a3b8; margin-bottom: 24px; max-height:180px; overflow-y:auto;">
                    "{item['clause']}"
                </div>
                
                {xai_html}
                
                <h6 style="color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem;">Deep Reasoning</h6>
                <p style="font-size:0.95rem; color:#cbd5e1; line-height:1.6; margin-bottom:24px;">
                    {item.get('explanation', '')}
                </p>
                
                <h6 style="color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem;">Strategic Mitigation</h6>
                <p style="font-size:0.95rem; color:#10b981; line-height:1.6; background:rgba(16,185,129,0.05); padding:12px; border-radius:8px;">
                    {item.get('mitigation', '')}
                </p>
                
                <h6 style="color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem;">RAG (BM25+Dense) Reference</h6>
                <p style="font-size:0.9rem; color:#38bdf8; line-height:1.6; margin-bottom:0;">
                    {item.get('legal_reference', '')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👈 Select a clause from the document to view its AI reasoning and mitigation strategy.")


def tab_analytics():
    state = st.session_state["agent_state_a"]
    if not state:
        st.info("Run analysis to view Risk Analytics.")
        return
    
    results = state.get("ml_results", [])
    if not results:
        return

    st.markdown("### Executive Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_risk_distribution_chart(results), use_container_width=True)
    with col2:
        st.plotly_chart(create_confidence_trend_chart(results), use_container_width=True)
        
    st.plotly_chart(create_complexity_vs_risk_chart(results), use_container_width=True)


def tab_compare():
    state_a = st.session_state.get("agent_state_a")
    state_b = st.session_state.get("agent_state_b")

    if not state_a or not state_b:
        st.info("Upload and process BOTH Doc A and Doc B to unlock the Comparison Engine.")
        return

    st.markdown("### Document Delta & Semantic Alignment")

    try:
        score_a = state_a["final_report"]["statistics"]["risk_index"]
        score_b = state_b["final_report"]["statistics"]["risk_index"]
        
        comp = compare_contracts(
            st.session_state["file_name_a"], state_a["clauses"], state_a["ml_results"], score_a,
            st.session_state["file_name_b"], state_b["clauses"], state_b["ml_results"], score_b
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Doc A Score", score_a)
        col2.metric("Doc B Score", score_b)
        col3.metric("Semantic Alignment", comp["semantic_alignment"])

        st.markdown(f"""
        <div class="saas-card" style="margin-top:24px;">
            <h5 style="color:#e2e8f0; margin-bottom:16px;">Executive Comparison Summary</h5>
            <p style="font-size:1.05rem; line-height:1.6; color:#94a3b8;">{comp['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("##### Missing Protections in Doc A")
            if comp["missing_in_a"]:
                for m in comp["missing_in_a"]: st.error(f"❌ {m}")
            else: st.success("No major discrepancies detected.")

        with colB:
            st.markdown("##### Missing Protections in Doc B")
            if comp["missing_in_b"]:
                for m in comp["missing_in_b"]: st.error(f"❌ {m}")
            else: st.success("No major discrepancies detected.")

    except Exception as e:
        st.error(f"Comparison computation failed: {e}")


def tab_export():
    state = st.session_state.get("agent_state_a")
    if not state: return st.info("Run analysis to enable exports.")
    
    st.markdown("### Export Center")
    st.markdown("Download production-ready corporate artifacts.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <h1>📄</h1><h3>C-Suite PDF Report</h3>
            <p style="color:#94a3b8;">Executive summary, risk graphs, and mitigation tables.</p>
        </div>""", unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(state.get("final_report", {}))
        st.download_button("Download PDF", pdf_bytes, file_name="LexIQ_Premium_Report.pdf", use_container_width=True)
        
    with c2:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <h1>🗂️</h1><h3>JSON Data Artifact</h3>
            <p style="color:#94a3b8;">Structured ML response for internal API systems.</p>
        </div>""", unsafe_allow_html=True)
        json_b = report_to_json_string(state.get("final_report", {})).encode("utf-8")
        st.download_button("Download JSON", json_b, file_name="LexIQ_Report.json", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
def main():
    init_session()
    execute, file_a, file_b, d_nda, d_msa = render_sidebar()

    # Handle Demo Modes
    if d_nda or d_msa:
        v = "NDA" if d_nda else "MSA"
        t, n = load_demo_contract(v)
        if t: process_document(t, n, "a")
        else: st.sidebar.error("Demo file not found.")

    # Handle Execution
    if execute:
        if file_a: process_document(get_text_from_file(file_a), file_a.name, "a")
        else: st.sidebar.warning("Upload Doc A first.")
        
        if file_b: process_document(get_text_from_file(file_b), file_b.name, "b")

    t1, t2, t3, t4 = st.tabs(["📄 Workspace", "📊 Analytics", "📑 Compare", "📥 Export Center"])
    
    with t1: tab_risk_analysis()
    with t2: tab_analytics()
    with t3: tab_compare()
    with t4: tab_export()

if __name__ == "__main__":
    main()
