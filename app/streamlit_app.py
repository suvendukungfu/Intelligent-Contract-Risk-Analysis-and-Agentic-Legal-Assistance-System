"""
app/streamlit_app.py  —  Milestone 5: Professional UI/UX SaaS Platform
======================================================================
Enterprise-grade Legal AI Platform.
"""

import os
import sys
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd

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
    page_title="LexIQ Enterprise AI", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    
    .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 12px; padding: 6px; box-shadow: inset 0px 2px 4px rgba(0,0,0,0.2); }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #94a3b8; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: #1e293b !important; color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }
    
    .saas-card {
        background: #1e293b; border-radius: 16px; padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5), 0 2px 4px -1px rgba(0,0,0,0.3);
        border: 1px solid #334155; margin-bottom: 24px; transition: transform 0.2s;
    }
    
    .metric-title { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 8px; }
    .metric-value { font-size: 2.5rem; color: #f8fafc; font-weight: 700; line-height: 1; }
    .metric-value.critical { color: #ef4444; }
    .metric-value.warning { color: #f59e0b; }
    .metric-value.safe { color: #10b981; }
    
    .badge { padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
    .badge-high { background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge-low { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .kw-chip { background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56,189,248,0.2); padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; margin: 2px; display: inline-block; }
    
    .footer { position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background: #0b0f19; color: #64748b; font-size: 0.8rem; border-top: 1px solid #1e293b; z-index: 1000;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE & HELPERS
# ══════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "agent_state_a": None, "file_name_a": "", "raw_a": "",
        "agent_state_b": None, "file_name_b": "", "raw_b": "",
        "selected_clause_idx": None,
        "trigger_execution": False,
        "demo_to_load": None
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def get_text_from_file(file) -> str:
    if file.name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
                if not text: raise ValueError("PDF contains no readable text.")
                return text
        except Exception as e:
            st.error(f"Please upload a valid PDF or TXT contract. (Error: {e})")
            return ""
    elif file.name.endswith(".txt"):
        return file.getvalue().decode("utf-8", errors="ignore")
    else:
        st.error("Invalid file format. Please upload a valid PDF or TXT contract.")
        return ""

def load_demo_contract(variant="NDA"):
    path = os.path.join(ROOT, "sample_docs", f"sample_{variant.lower()}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), f"LexIQ_Demo_{variant}.txt"
    return "", ""

def process_document(raw_text: str, filename: str, doc_key: str):
    if not raw_text.strip(): return
    with st.spinner(f"Analyzing {filename}... Parse → Identify → RAG → LLM Reasoning"):
        state = run_agent_pipeline(raw_text, file_name=filename)
        st.session_state[f"agent_state_{doc_key}"] = state
        st.session_state[f"file_name_{doc_key}"] = filename
        st.session_state[f"raw_{doc_key}"] = raw_text

# ══════════════════════════════════════════════════════════════════
# ONBOARDING / EMPTY STATE
# ══════════════════════════════════════════════════════════════════
def render_onboarding():
    st.markdown("<h1 style='text-align:center; margin-bottom: 24px;'>Welcome to your AI Legal Assistant</h1>", unsafe_allow_html=True)
    
    with st.expander("📖 How to Use This AI Legal Assistant", expanded=True):
        st.markdown("""
        **Welcome to the fastest way to review legal documents.**
        1. 📤 **Upload** a contract (PDF or TXT) in the sidebar.
        2. 🚨 **View** clause-level risk analysis identifying predatory traps.
        3. 🧠 **Click** "Explain This Clause" for detailed AI reasoning and mitigation strategies.
        4. 📊 **Review** executive-level dashboard analytics.
        5. 📥 **Download** the generated PDF report.
        """)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<div class='saas-card' style='text-align:center;'>", unsafe_allow_html=True)
        st.markdown("### First time here?")
        st.markdown("Experience the full power of agentic legal review instantly.")
        if st.button("🚀 Try Demo Contract (NDA)", use_container_width=True, help="Loads a pre-configured sample contract to demonstrate capabilities."):
            st.session_state["demo_to_load"] = "NDA"
            st.session_state["trigger_execution"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN VIEW RENDERING
# ══════════════════════════════════════════════════════════════════
def render_kpi_bar(stats):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="saas-card" style="padding:16px;" title="Indicates overall contract risk level out of 10">
            <div class="metric-title">Risk Score</div>
            <div class="metric-value { 'critical' if stats.get('risk_index',0) >= 7 else 'safe'}">{stats.get('risk_index', '0.0')}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="saas-card" style="padding:16px;" title="Total number of critical clauses needing human review">
            <div class="metric-title">High Risk Count</div>
            <div class="metric-value critical">{stats.get('high_risk_clauses', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="saas-card" style="padding:16px;" title="Total number of discrete provisions segmented">
            <div class="metric-title">Total Clauses</div>
            <div class="metric-value">{stats.get('total_clauses', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="saas-card" style="padding:16px;" title="Model certainty level">
            <div class="metric-title">Avg Confidence</div>
            <div class="metric-value" style="color:#38bdf8;">{stats.get('avg_confidence', '0%')}</div>
        </div>""", unsafe_allow_html=True)


def render_xai_block(item):
    xai_html = ""
    if item.get("xai_weights"):
        xai_html = "<h6 style='color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem; margin-top:16px;'>Mathematical Feature Importance</h6>"
        for word, weight in item.get("xai_weights").items():
            w_pct = min(weight * 100, 100)
            xai_html += f"""
            <div style="display:flex; align-items:center; margin-bottom:4px;" title="The presence of '{word}' drove risk probability up by {weight:.2f}">
                <div style="width:100px; color:#94a3b8; font-size:0.8rem; overflow:hidden; text-overflow:ellipsis;">{word}</div>
                <div style="flex-grow:1; background:#1f2937; height:8px; border-radius:4px; margin:0 10px;">
                    <div style="width:{w_pct}%; background:#ef4444; height:100%; border-radius:4px;"></div>
                </div>
                <div style="font-size:0.75rem; color:#f87171;">+{weight:.2f}</div>
            </div>
            """
        xai_html += "<div style='margin-bottom:24px;'></div>"
    return xai_html


def tab_risk_analysis():
    state = st.session_state["agent_state_a"]
    stats = state["final_report"]["statistics"]
    risks = state["final_report"]["identified_risks"]

    st.markdown("### 📄 Clause-Level Risk Analysis")
    render_kpi_bar(stats)

    if not risks:
        st.success("No high-risk clauses found. This contract appears mathematically safe based on the trained model parameters.")
        return

    st.markdown("Select a clause below to view detailed breakdown.")
    for idx, item in enumerate(risks):
        r_level = item.get("risk_level")
        chips = "".join([f"<span class='kw-chip'>{t}</span>" for t in item.get("linguistic_triggers", [])])
        if item.get("is_anomaly"):
            chips += f" <span class='kw-chip' style='background:rgba(217,70,239,0.1);color:#d946ef;border-color:#d946ef'>🧬 ANOMALY</span>"
            
        st.markdown(f"""
        <div style="background:#1e293b; padding:16px; border-radius:8px; border-left:4px solid {'#ef4444' if r_level=='High Risk' else '#10b981'}; margin-bottom:8px;">
            <div style="margin-bottom:8px;">
                <span class="badge {'badge-high' if r_level == 'High Risk' else 'badge-low'}">{r_level}</span>
                <span style="font-size:0.8rem; color:#94a3b8; margin-left:8px;">Confidence: {item['confidence']}</span>
                {chips}
            </div>
            <div style="font-style:italic; font-size:0.9rem; color:#cbd5e1; margin-bottom: 12px;">"{item['clause'][:150]}..."</div>
        </div>
        """, unsafe_allow_html=True)
        
        # UI Button to jump to specific clause explanation
        if st.button("🧠 Explain This Clause", key=f"btn_{idx}", help="Get Deep AI reasoning and mitigation for this specific clause"):
            st.session_state["selected_clause_idx"] = idx
            st.toast("To view the deep explanation, switch to the '🧠 AI Legal Assistant' tab.")

def tab_ai_assistant():
    state = st.session_state["agent_state_a"]
    risks = state["final_report"]["identified_risks"]
    idx = st.session_state.get("selected_clause_idx")

    st.markdown("### 🧠 AI Legal Reasoning Engine")
    
    if idx is None or idx >= len(risks):
        st.info("👈 Please select 'Explain This Clause' from the Risk Analysis tab to view the breakdown here.")
        return

    item = risks[idx]
    conf_float = float(item.get("confidence", "0").replace("%", ""))
    
    warning_html = ""
    if conf_float < 75.0:
        warning_html = "<div style='color:#f59e0b; background:rgba(245,158,11,0.1); padding:12px; border-radius:6px; font-size:0.85rem; margin-bottom:16px;'>⚠️ <b>Low Confidence Warning:</b> The AI is uncertain. Human review strongly advised.</div>"

    xai_html = render_xai_block(item)

    st.markdown(f"""
    <div class="saas-card" style="margin-bottom:0;">
        {warning_html}
        <h6 style="color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem;">Full Target Clause</h6>
        <div style="background:#0b0f19; padding:16px; border-radius:8px; font-style:italic; font-size:0.9rem; color:#94a3b8; margin-bottom: 24px; max-height:200px; overflow-y:auto;">
            "{item['clause']}"
        </div>
        
        {xai_html}
        
        <h6 style="color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem;">Deep Reasoning</h6>
        <p style="font-size:0.95rem; color:#cbd5e1; line-height:1.6; margin-bottom:24px; background:#1e293b; padding:16px; border-radius:8px;">
            {item.get('explanation', '')}
        </p>
        
        <h6 style="color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem;">Strategic Mitigation</h6>
        <p style="font-size:0.95rem; color:#10b981; line-height:1.6; background:rgba(16,185,129,0.05); padding:16px; border-radius:8px; margin-bottom:24px;">
            {item.get('mitigation', '')}
        </p>
        
        <h6 style="color:#e2e8f0; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem;">RAG Reference Case Law / Knowledge</h6>
        <p style="font-size:0.9rem; color:#38bdf8; line-height:1.6; background:rgba(56,189,248,0.05); padding:16px; border-radius:8px; margin-bottom:0;">
            {item.get('legal_reference', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)


def tab_analytics():
    state = st.session_state["agent_state_a"]
    st.markdown("### 📊 Enterprise Analytics")
    st.markdown("Monitor risk trends and model confidence distributions across the parsed document.")
    
    results = state.get("ml_results", [])
    if not results: return

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(create_risk_distribution_chart(results), use_container_width=True)
    with col2: st.plotly_chart(create_confidence_trend_chart(results), use_container_width=True)
    st.plotly_chart(create_complexity_vs_risk_chart(results), use_container_width=True)


def tab_compare():
    st.markdown("### 📑 Multi-Contract Compare")
    st.markdown("Compute the semantic delta between the primary document and a benchmark template.")
    
    state_a = st.session_state.get("agent_state_a")
    state_b = st.session_state.get("agent_state_b")

    if not state_a or not state_b:
        st.info("Upload and execute Document A and Document B in the sidebar to activate.")
        return

    try:
        score_a = state_a["final_report"]["statistics"]["risk_index"]
        score_b = state_b["final_report"]["statistics"]["risk_index"]
        comp = compare_contracts(
            st.session_state["file_name_a"], state_a["clauses"], state_a["ml_results"], score_a,
            st.session_state["file_name_b"], state_b["clauses"], state_b["ml_results"], score_b
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Doc A Risk Score", score_a, help="Risk Index for the primary document")
        col2.metric("Doc B Risk Score", score_b, help="Risk Index for the baseline document")
        col3.metric("Semantic Alignment", comp["semantic_alignment"], help="Structural and semantic similarity percentage")

        st.markdown(f"""
        <div class="saas-card" style="margin-top:24px;">
            <h5 style="color:#e2e8f0; margin-bottom:16px;">Executive Delta Summary</h5>
            <p style="font-size:1.05rem; line-height:1.6; color:#94a3b8;">{comp['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("##### Missing Protections in Doc A")
            if comp["missing_in_a"]:
                for m in comp["missing_in_a"]: st.error(f"❌ {m}")
            else: st.success("No discrepancies detected.")
        with colB:
            st.markdown("##### Missing Protections in Doc B")
            if comp["missing_in_b"]:
                for m in comp["missing_in_b"]: st.error(f"❌ {m}")
            else: st.success("No discrepancies detected.")

    except Exception as e:
        st.error(f"Comparison computation failed: {e}")


def tab_export():
    state = st.session_state["agent_state_a"]
    st.markdown("### 📥 Download Reports")
    st.markdown("Generate presentation-ready security audits and programmatic artifacts.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <h1>📄</h1><h3>C-Suite PDF Report</h3>
        </div>""", unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(state.get("final_report", {}))
        st.download_button("📥 Download PDF", pdf_bytes, file_name="LexIQ_Premium_Report.pdf", use_container_width=True, help="Download securely formatted PDF report for executives")
        
    with c2:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <h1>🗂️</h1><h3>JSON Data Artifact</h3>
        </div>""", unsafe_allow_html=True)
        json_b = report_to_json_string(state.get("final_report", {})).encode("utf-8")
        st.download_button("📥 Download JSON", json_b, file_name="LexIQ_Report.json", use_container_width=True, help="Raw AI structured output for API integration via JSON")


# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
def main():
    init_session()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding-bottom: 12px;">
            <div style="font-size:2.5rem; line-height:1;">⚖️</div>
            <div style="font-size:1.5rem; font-weight:800; color:#f8fafc;">LexIQ SaaS</div>
        </div>
        """, unsafe_allow_html=True)
        
        file_a = st.file_uploader("Primary Contract", type=["pdf", "txt"], key="upload_a", help="Upload your contract here (TXT or PDF max 5MB).")
        file_b = st.file_uploader("Baseline Reference (Optional)", type=["pdf", "txt"], key="upload_b", help="Upload a separate document to run semantic variance delta comparison.")
        
        c1, c2 = st.columns(2)
        btn_d1 = c1.button("Demo NDA", help="Preload a sample NDA")
        btn_d2 = c2.button("Demo MSA", help="Preload a sample MSA")
        
        if btn_d1: st.session_state["demo_to_load"] = "NDA"
        if btn_d2: st.session_state["demo_to_load"] = "MSA"
        
        execute = st.button("🚀 Analyze Contract", use_container_width=True, type="primary", help="Trigger the Machine Learning analysis pipeline.")
        if execute: st.session_state["trigger_execution"] = True

    # Handle Pending Executions (From Sidebar OR Onboarding Quick-Action)
    if st.session_state["trigger_execution"]:
        st.session_state["trigger_execution"] = False
        
        if st.session_state["demo_to_load"]:
            t, n = load_demo_contract(st.session_state["demo_to_load"])
            if t: process_document(t, n, "a")
            st.session_state["demo_to_load"] = None
        else:
            if file_a: 
                t = get_text_from_file(file_a)
                if t: process_document(t, file_a.name, "a")
            else: 
                st.toast("Please upload a contract or select a Demo file.", icon="⚠️")
            
            if file_b:
                tb = get_text_from_file(file_b)
                if tb: process_document(tb, file_b.name, "b")

    # Render Content
    if not st.session_state["agent_state_a"]:
        render_onboarding()
    else:
        # Full App with 5 Tabs
        t1, t2, t3, t4, t5 = st.tabs(["📄 Risk Analysis", "🧠 AI Legal Assistant", "📊 Analytics", "📑 Compare Contracts", "📥 Export Report"])
        
        with t1: tab_risk_analysis()
        with t2: tab_ai_assistant()
        with t3: tab_analytics()
        with t4: tab_compare()
        with t5: tab_export()

    # Sticky Professional Footer
    st.markdown('<div class="footer">LexIQ Intelligent Platform • AI-generated analysis. Not a substitute for professional legal advice.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
