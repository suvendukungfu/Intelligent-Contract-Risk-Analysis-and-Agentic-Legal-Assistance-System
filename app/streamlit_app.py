"""
app/streamlit_app.py  —  Milestone 6: B2B Enterprise UI Override
======================================================================
Minimalist, high-end, zero-emoji SaaS Legal AI Platform.
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
    page_title="LEXIQ | ENTERPRISE", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif; }
    
    /* Deep Cyber-Indigo Atmosphere */
    .stApp { background: linear-gradient(135deg, #09090b 0%, #0c0a20 100%); color: #a1a1aa; }
    [data-testid="stSidebar"] { background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(20px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    
    /* Strict Tab Styling */
    .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid rgba(255,255,255,0.1); padding: 0; gap: 32px; }
    .stTabs [data-baseweb="tab"] { color: #71717a; font-weight: 600; padding: 12px 0; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 1px; border: none !important; background: transparent !important; }
    .stTabs [aria-selected="true"] { color: #e0e7ff !important; border-bottom: 2px solid #818cf8 !important; text-shadow: 0 0 10px rgba(129, 140, 248, 0.5); }
    
    /* Glassmorphism Cards */
    .saas-card {
        background: rgba(30, 30, 40, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px; 
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05); 
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
    }
    
    /* Glowing Metrics */
    .metric-title { font-size: 0.75rem; color: #818cf8; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 12px; font-family: 'Outfit', sans-serif; }
    .metric-value { font-size: 2.25rem; color: #f4f4f5; font-weight: 500; line-height: 1; font-family: 'JetBrains Mono', monospace; letter-spacing: -1px; text-shadow: 0 0 15px rgba(255,255,255,0.2); }
    .metric-value.critical { color: #f87171; text-shadow: 0 0 15px rgba(248, 113, 113, 0.4); }
    .metric-value.warning { color: #fbbf24; text-shadow: 0 0 15px rgba(251, 191, 36, 0.4); }
    .metric-value.safe { color: #34d399; text-shadow: 0 0 15px rgba(52, 211, 153, 0.4); }
    
    /* Hollow Badges */
    .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; display: inline-block; font-family: 'Outfit', sans-serif; box-shadow: inset 0 0 10px rgba(0,0,0,0.5); }
    .badge-high { color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.5); background: rgba(248, 113, 113, 0.1); }
    .badge-low { color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.5); background: rgba(52, 211, 153, 0.1); }
    .kw-chip { color: #c7d2fe; border: 1px solid rgba(129, 140, 248, 0.3); padding: 3px 8px; border-radius: 4px; font-size: 0.65rem; margin: 2px; display: inline-block; text-transform: uppercase; letter-spacing: 1px; background: rgba(129, 140, 248, 0.05); }
    
    .footer { position: fixed; bottom: 0; width: 100%; text-align: center; padding: 12px; background: rgba(0,0,0,0.8); backdrop-filter: blur(10px); color: #52525b; font-size: 0.75rem; letter-spacing: 0.5px; border-top: 1px solid rgba(255,255,255,0.05); z-index: 1000;}
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
                if not text: raise ValueError("PDF payload empty.")
                return text
        except Exception as e:
            st.error(f"[ERR_FILE_PARSE] Critical failure reading PDF. Exception: {e}")
            return ""
    elif file.name.endswith(".txt"):
        return file.getvalue().decode("utf-8", errors="ignore")
    else:
        st.error("[ERR_TYPE_MISMATCH] Strict validation: Only PDF/TXT permitted.")
        return ""

def load_demo_contract(variant="NDA"):
    path = os.path.join(ROOT, "sample_docs", f"sample_{variant.lower()}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), f"LEXIQ_BENCHMARK_{variant}.txt"
    return "", ""

def process_document(raw_text: str, filename: str, doc_key: str):
    if not raw_text.strip(): return
    with st.spinner(f"PROCESSING // {filename} // NLP MAP → INFERENCE → RAG"):
        state = run_agent_pipeline(raw_text, file_name=filename)
        st.session_state[f"agent_state_{doc_key}"] = state
        st.session_state[f"file_name_{doc_key}"] = filename
        st.session_state[f"raw_{doc_key}"] = raw_text

# ══════════════════════════════════════════════════════════════════
# ONBOARDING / EMPTY STATE
# ══════════════════════════════════════════════════════════════════
def render_onboarding():
    st.markdown("<h1 style='text-align:center; color:#f4f4f5; font-weight:300; letter-spacing: -1px; margin-bottom: 8px;'>LEXIQ AI ASSISTANT</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#71717a; font-weight:400; margin-bottom:48px; letter-spacing: 0.5px;'>CONTRACT RISK ANALYSIS PLATFORM</h4>", unsafe_allow_html=True)
    
    with st.expander("[ INSTRUCTION PROTOCOL ]", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.markdown("""
            **SYSTEM CAPABILITIES:**
            * [x] Detect adversarial provisions
            * [x] Synthesize legal context
            * [x] Formulate strategic mitigation
            * [x] Generate compliance artifacts
            
            **EXECUTION WORKFLOW:**
            1. Mount document payload (Sidebar)
            2. Review Risk Analysis matrix
            3. Execute 'EXPLAIN' for high-confidence anomalies
            4. Review AI mitigation heuristics
            5. Export final artifact
            """)
        with colB:
            st.markdown("""
            **TECHNICAL ARCHITECTURE:**
            * **NLP Core:** spaCy Structural Parser
            * **Inference:** TF-IDF + Logistic Regressor
            * **Orchestration:** LangGraph Agents
            * **RAG Engine:** ChromaDB Vector Store
            * **LLM Engine:** Generative Synthesizer
            """)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1.5,2,1.5])
    with c2:
        st.markdown("<div class='saas-card' style='text-align:center;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#f4f4f5; font-weight:400; margin-bottom:8px;'>SYSTEM INITIALIZATION</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#71717a; font-size:0.9rem; margin-bottom:24px;'>Awaiting payload or select benchmark benchmark below.</p>", unsafe_allow_html=True)
        if st.button("► EXECUTE BENCHMARK (NDA)", use_container_width=True, help="Load pre-configured test dataset"):
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
        st.markdown(f"""<div class="saas-card" title="Calculated system-wide severity coefficient">
            <div class="metric-title">RISK QUOTIENT</div>
            <div class="metric-value { 'critical' if stats.get('risk_index',0) >= 7 else 'safe'}">{stats.get('risk_index', '0.0')}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="saas-card" title="Count of provisions exceeding safety thresholds">
            <div class="metric-title">CRITICAL FLAGS</div>
            <div class="metric-value critical">{stats.get('high_risk_clauses', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="saas-card" title="Total segmented elements analyzed">
            <div class="metric-title">CLAUSES PARSED</div>
            <div class="metric-value">{stats.get('total_clauses', 0)}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="saas-card" title="Algorithm predictive certainty avg">
            <div class="metric-title">MODEL CERTAINTY</div>
            <div class="metric-value" style="color:#71717a;">{stats.get('avg_confidence', '0%')}</div>
        </div>""", unsafe_allow_html=True)


def render_xai_block(item):
    xai_html = ""
    if item.get("xai_weights"):
        xai_html = "<h6 style='color:#a1a1aa; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.75rem; margin-top:24px; margin-bottom:12px;'>[ XAI FEATURE IMPORTANCE ]</h6>"
        for word, weight in item.get("xai_weights").items():
            w_pct = min(weight * 100, 100)
            xai_html += f"""
            <div style="display:flex; align-items:center; margin-bottom:8px;" title="Token '{word}' regression coefficient: +{weight:.2f}">
                <div style="width:120px; color:#71717a; font-size:0.75rem; font-family:'JetBrains Mono', monospace;">[ {word.upper()} ]</div>
                <div style="flex-grow:1; background:#18181b; height:2px; margin:0 12px; position:relative;">
                    <div style="width:{w_pct}%; background:#f87171; height:100%; position:absolute; left:0;"></div>
                </div>
                <div style="font-size:0.75rem; color:#f87171; font-family:'JetBrains Mono', monospace;">+{weight:.2f}</div>
            </div>
            """
        xai_html += "<div style='margin-bottom:32px;'></div>"
    return xai_html


def tab_risk_analysis():
    state = st.session_state["agent_state_a"]
    stats = state["final_report"]["statistics"]
    risks = state["final_report"]["identified_risks"]

    st.markdown("<h3 style='font-weight:400; color:#f4f4f5; letter-spacing:-0.5px;'>RISK ANALYSIS MATRIX</h3>", unsafe_allow_html=True)
    render_kpi_bar(stats)

    if not risks:
        st.markdown("<div style='border-left:2px solid #34d399; padding-left:16px; color:#34d399; font-family:monospace;'>[SYS_OK] Zero high-risk provisions detected. Document conforms to safety baseline.</div>", unsafe_allow_html=True)
        return

    st.info("[NEXT STEP REQUIRED] Select '► EXPLAIN PROVISION' below to activate the AI Assistant and generate deep reasoning context.")
    st.markdown("<div style='font-size:0.8rem; color:#71717a; text-transform:uppercase; margin-bottom:16px;'>PROVISION BREAKDOWN:</div>", unsafe_allow_html=True)
    for idx, item in enumerate(risks):
        r_level = item.get("risk_level")
        chips = "".join([f"<span class='kw-chip'>{t}</span>" for t in item.get("linguistic_triggers", [])])
        if item.get("is_anomaly"):
            chips += f" <span class='kw-chip' style='color:#c084fc; border-color:#c084fc;'>[ ANOMALY SCORE: {item.get('anomaly_score')} ]</span>"
            
        st.markdown(f"""
        <div style="background:#000000; padding:20px; border:1px solid #27272a; margin-bottom:12px;">
            <div style="margin-bottom:12px; display:flex; align-items:center;">
                <span class="badge {'badge-high' if r_level == 'High Risk' else 'badge-low'}">{r_level}</span>
                <span style="font-size:0.75rem; color:#52525b; margin-left:16px; font-family:monospace;">CONFIDENCE: {item['confidence']}</span>
                <div style="margin-left:auto;">{chips}</div>
            </div>
            <div style="font-size:0.85rem; color:#a1a1aa; line-height:1.6;">{item['clause'][:200]}...</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("► EXPLAIN PROVISION", key=f"btn_{idx}", help="Extract deep AI reasoning logic"):
            st.session_state["selected_clause_idx"] = idx
            st.toast("Accessing reasoning engine -> See AI ASSISTANT tab.", icon="●")

def tab_ai_assistant():
    state = st.session_state["agent_state_a"]
    risks = state["final_report"]["identified_risks"]
    idx = st.session_state.get("selected_clause_idx")

    st.markdown("<h3 style='font-weight:400; color:#f4f4f5; letter-spacing:-0.5px;'>AI REASONING ENGINE</h3>", unsafe_allow_html=True)
    
    if idx is None or idx >= len(risks):
        st.info("[AWAITING INPUT] Select '► EXPLAIN PROVISION' in Risk Analysis to mount data.")
        return

    item = risks[idx]
    conf_float = float(item.get("confidence", "0").replace("%", ""))
    
    warning_html = ""
    if conf_float < 75.0:
        warning_html = "<div style='color:#fbbf24; border-left:2px solid #fbbf24; padding-left:12px; font-size:0.8rem; margin-bottom:24px; font-family:monospace;'>[WARNING] Low prediction confidence mapped. Review mandatory.</div>"

    xai_html = render_xai_block(item)

    st.markdown(f"""
    <div class="saas-card" style="margin-bottom:0;">
        {warning_html}
        <h6 style="color:#71717a; font-family:'JetBrains Mono', monospace; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.7rem; margin-bottom:8px;">[ TARGET PAYLOAD ]</h6>
        <div style="background:#000000; padding:16px; border:1px solid #18181b; font-size:0.85rem; color:#a1a1aa; margin-bottom: 32px; max-height:200px; overflow-y:auto;">
            {item['clause']}
        </div>
        
        {xai_html}
        
        <h6 style="color:#71717a; font-family:'JetBrains Mono', monospace; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.7rem; margin-bottom:8px;">[ REASONING VECTOR ]</h6>
        <p style="font-size:0.9rem; color:#e4e4e7; line-height:1.6; margin-bottom:32px; border-left:1px solid #3f3f46; padding-left:16px;">
            {item.get('explanation', '')}
        </p>
        
        <h6 style="color:#71717a; font-family:'JetBrains Mono', monospace; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.7rem; margin-bottom:8px;">[ STRATEGIC MITIGATION ]</h6>
        <p style="font-size:0.9rem; color:#34d399; line-height:1.6; border-left:1px solid #34d399; padding-left:16px; margin-bottom:32px;">
            {item.get('mitigation', '')}
        </p>
        
        <h6 style="color:#71717a; font-family:'JetBrains Mono', monospace; font-weight:600; text-transform:uppercase; letter-spacing:1px; font-size:0.7rem; margin-bottom:8px;">[ RAG SOURCE KNOWLEDGE ]</h6>
        <p style="font-size:0.85rem; color:#38bdf8; line-height:1.6; border-left:1px solid #38bdf8; padding-left:16px; margin-bottom:0;">
            {item.get('legal_reference', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)


def tab_analytics():
    state = st.session_state["agent_state_a"]
    st.markdown("<h3 style='font-weight:400; color:#f4f4f5; letter-spacing:-0.5px;'>TELEMETRY DASHBOARD</h3>", unsafe_allow_html=True)
    
    results = state.get("ml_results", [])
    if not results: return

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(create_risk_distribution_chart(results), use_container_width=True)
    with col2: st.plotly_chart(create_confidence_trend_chart(results), use_container_width=True)
    st.plotly_chart(create_complexity_vs_risk_chart(results), use_container_width=True)


def tab_compare():
    st.markdown("<h3 style='font-weight:400; color:#f4f4f5; letter-spacing:-0.5px;'>DELTA SEQUENCE: COMPARISON</h3>", unsafe_allow_html=True)
    
    state_a = st.session_state.get("agent_state_a")
    state_b = st.session_state.get("agent_state_b")

    if not state_a or not state_b:
        st.info("[AWAITING CONTEXT] Establish Doc A and Doc B payload to run comparison.")
        return

    try:
        score_a = state_a["final_report"]["statistics"]["risk_index"]
        score_b = state_b["final_report"]["statistics"]["risk_index"]
        comp = compare_contracts(
            st.session_state["file_name_a"], state_a["clauses"], state_a["ml_results"], score_a,
            st.session_state["file_name_b"], state_b["clauses"], state_b["ml_results"], score_b
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("NODE A RISK", score_a)
        col2.metric("NODE B RISK", score_b)
        col3.metric("STRUCTURAL ALIGNMENT", comp["semantic_alignment"])

        st.markdown(f"""
        <div class="saas-card" style="margin-top:32px;">
            <h6 style="color:#71717a; font-family:'JetBrains Mono', monospace; font-weight:600; text-transform:uppercase; margin-bottom:16px;">[ ALIGNMENT SUMMARY ]</h6>
            <p style="font-size:0.95rem; line-height:1.6; color:#a1a1aa;">{comp['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("<h6 style='color:#71717a; font-family:monospace; margin-bottom:16px;'>[ MISSING IN DOC A ]</h6>", unsafe_allow_html=True)
            if comp["missing_in_a"]:
                for m in comp["missing_in_a"]: st.error(f"● {m}")
            else: st.success("● Null discrepancies.")
        with colB:
            st.markdown("<h6 style='color:#71717a; font-family:monospace; margin-bottom:16px;'>[ MISSING IN DOC B ]</h6>", unsafe_allow_html=True)
            if comp["missing_in_b"]:
                for m in comp["missing_in_b"]: st.error(f"● {m}")
            else: st.success("● Null discrepancies.")

    except Exception as e:
        st.error(f"[ERR_COMPARISON] Fault: {e}")


def tab_export():
    state = st.session_state["agent_state_a"]
    st.markdown("<h3 style='font-weight:400; color:#f4f4f5; letter-spacing:-0.5px;'>ARTIFACT EXTRACTION</h3>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.5rem; margin-bottom:16px; color:#f4f4f5;">[ PDF ]</div>
            <h4 style="color:#a1a1aa; font-weight:400; margin-bottom:24px;">EXEC_SUMMARY.PDF</h4>
            <div style="border-top:1px solid #27272a; padding-top:16px; margin-bottom:16px; font-size:0.8rem; color:#71717a;">High-fidelity vector graphic report tailored for executive review.</div>
        </div>""", unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(state.get("final_report", {}))
        st.download_button("[ DOWNLOAD INSTANCE ]", pdf_bytes, file_name="LexIQ_Audit.pdf", use_container_width=True, help="Extract PDF block")
        
    with c2:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.5rem; margin-bottom:16px; color:#f4f4f5;">[ JSON ]</div>
            <h4 style="color:#a1a1aa; font-weight:400; margin-bottom:24px;">PAYLOAD.JSON</h4>
            <div style="border-top:1px solid #27272a; padding-top:16px; margin-bottom:16px; font-size:0.8rem; color:#71717a;">Raw, structured machine learning data arrays for API or internal DB ingress.</div>
        </div>""", unsafe_allow_html=True)
        json_b = report_to_json_string(state.get("final_report", {})).encode("utf-8")
        st.download_button("[ DOWNLOAD INSTANCE ]", json_b, file_name="LexIQ_Data.json", use_container_width=True, help="Extract JSON block")


# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
def main():
    init_session()
    
    with st.sidebar:
        st.markdown("""
        <div style="padding-bottom: 32px; border-bottom: 1px solid #18181b; margin-bottom:24px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size:1.2rem; font-weight:500; color:#f4f4f5; letter-spacing: 2px;">LEXIQ.SYS</div>
            <div style="font-size:0.6rem; color:#71717a; letter-spacing: 1px;">ENTERPRISE COMPLIANCE V4.0</div>
        </div>
        """, unsafe_allow_html=True)
        
        file_a = st.file_uploader("TARGET PAYLOAD (DOC A)", type=["pdf", "txt"], key="upload_a", help="Upload core file (TXT/PDF)")
        file_b = st.file_uploader("BASELINE PAYLOAD (DOC B)", type=["pdf", "txt"], key="upload_b", help="Upload baseline standard")
        
        c1, c2 = st.columns(2)
        btn_d1 = c1.button("► MOUNT NDA", help="Preload benchmark")
        btn_d2 = c2.button("► MOUNT MSA", help="Preload benchmark")
        
        if btn_d1: st.session_state["demo_to_load"] = "NDA"
        if btn_d2: st.session_state["demo_to_load"] = "MSA"
        
        st.markdown("<br/>", unsafe_allow_html=True)
        execute = st.button("[ INITIALIZE ANALYSIS ]", use_container_width=True, type="primary", help="Trigger Neural Network Inference")
        if execute: st.session_state["trigger_execution"] = True

    # Handle Pending Executions
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
                st.toast("Please upload a contract or select a Demo file.", icon="●")
            
            if file_b:
                tb = get_text_from_file(file_b)
                if tb: process_document(tb, file_b.name, "b")

    # Render Base Content
    if not st.session_state["agent_state_a"]:
        render_onboarding()
    else:
        # Full App Nav
        t1, t2, t3, t4, t5 = st.tabs(["RISK ANALYSIS", "AI ASSISTANT", "ANALYTICS", "COMPARE", "EXPORT"])
        
        with t1: tab_risk_analysis()
        with t2: tab_ai_assistant()
        with t3: tab_analytics()
        with t4: tab_compare()
        with t5: tab_export()

    # Footer
    st.markdown('<div class="footer">LEXIQ.SYS | AI-GENERATED DIAGNOSTIC DATA. NOT LEGAL COUNSEL. DO NOT DISTRIBUTE WITHOUT HUMAN REVIEW.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
