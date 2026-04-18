"""
app/streamlit_app.py  —  Guided Story-Driven SaaS UX
======================================================================
Deep-Tech Glassmorphism combined with a linear, guided user journey.
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
# PAGE SETTINGS & GLOBAL CSS (Glassmorphism)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LexIQ - AI Legal Assistant", layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif; }
    
    /* Deep Cyber Atmosphere */
    .stApp { background: linear-gradient(135deg, #09090b 0%, #0f0c29 100%); color: #a1a1aa; }
    [data-testid="stSidebar"] { background: rgba(0, 0, 0, 0.4); backdrop-filter: blur(20px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    
    /* Tab Styling */
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
    .metric-title { font-size: 0.8rem; color: #818cf8; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 8px; font-family: 'Outfit', sans-serif; }
    .metric-value { font-size: 2.5rem; color: #f4f4f5; font-weight: 600; line-height: 1; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 15px rgba(255,255,255,0.2); }
    .metric-value.critical { color: #f87171; text-shadow: 0 0 15px rgba(248, 113, 113, 0.4); }
    .metric-value.warning { color: #fbbf24; text-shadow: 0 0 15px rgba(251, 191, 36, 0.4); }
    .metric-value.safe { color: #34d399; text-shadow: 0 0 15px rgba(52, 211, 153, 0.4); }
    
    /* Dynamic Stepper Bar */
    .stepper { display: flex; justify-content: space-between; margin-bottom: 32px; background: rgba(0,0,0,0.3); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); }
    .step { font-size: 0.8rem; text-transform: uppercase; font-family: 'Outfit'; font-weight: 600; color: #52525b; text-align: center; flex-grow: 1; position: relative; }
    .step.active { color: #818cf8; text-shadow: 0 0 8px rgba(129, 140, 248, 0.4); }
    .step.done { color: #34d399; }
    .step:not(:last-child)::after { content: '➔'; position: absolute; right: -10px; color: #3f3f46; font-size: 0.9rem; }
    
    .badge { padding: 6px 12px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; display: inline-block; font-family: 'Outfit', sans-serif; box-shadow: inset 0 0 10px rgba(0,0,0,0.5); }
    .badge-high { color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.5); background: rgba(248, 113, 113, 0.1); }
    .badge-low { color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.5); background: rgba(52, 211, 153, 0.1); }
    .kw-chip { color: #c7d2fe; border: 1px solid rgba(129, 140, 248, 0.3); padding: 3px 8px; border-radius: 4px; font-size: 0.65rem; margin: 2px; display: inline-block; background: rgba(129, 140, 248, 0.05); }
    
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
        "demo_to_load": None,
        "current_step": 1
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
            st.error(f"Failed reading PDF: {e}")
            return ""
    elif file.name.endswith(".txt"):
        return file.getvalue().decode("utf-8", errors="ignore")
    return ""

def load_demo_contract(variant="NDA"):
    path = os.path.join(ROOT, "sample_docs", f"sample_{variant.lower()}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), f"LexIQ_Demo_{variant}.txt"
    return "", ""

def process_document(raw_text: str, filename: str, doc_key: str):
    if not raw_text.strip(): return
    with st.spinner(f"Analyzing contract using AI models..."):
        state = run_agent_pipeline(raw_text, file_name=filename)
        st.session_state[f"agent_state_{doc_key}"] = state
        st.session_state[f"file_name_{doc_key}"] = filename
        st.session_state[f"raw_{doc_key}"] = raw_text
        st.session_state["current_step"] = 2

# ══════════════════════════════════════════════════════════════════
# DYNAMIC STEPPER
# ══════════════════════════════════════════════════════════════════
def render_stepper():
    step = st.session_state["current_step"]
    
    def cls(s):
        if step > s: return "done"
        if step == s: return "active"
        return ""
        
    html = f"""
    <div class="stepper">
        <div class="step {cls(1)}">1. Upload</div>
        <div class="step {cls(2)}">2. Analyze Risk</div>
        <div class="step {cls(3)}">3. Explain Cause</div>
        <div class="step {cls(4)}">4. Insights</div>
        <div class="step {cls(5)}">5. Report</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ONBOARDING / HERO SECTION
# ══════════════════════════════════════════════════════════════════
def render_onboarding():
    st.session_state["current_step"] = 1
    render_stepper()
    
    st.markdown("<h1 style='text-align:center; color:#f4f4f5; font-size:3.5rem; letter-spacing:-1px; margin-bottom: 8px;'>LexIQ</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#818cf8; font-weight:400; font-family:Outfit; margin-bottom:8px;'>AI Legal Risk Intelligence Platform</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#a1a1aa; font-size:1.1rem; margin-bottom:48px;'>Upload contracts. Detect risks. Understand legal impact. Take action.</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='saas-card' style='text-align:center; height:100%;'>
            <h3 style='margin-bottom:12px;'></h3>
            <h4 style='color:#f4f4f5;'>Risk Detection</h4>
            <p style='color:#71717a; font-size:0.9rem;'>Using Machine Learning to instantly identify toxic liability traps and unbalanced clauses.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='saas-card' style='text-align:center; height:100%;'>
            <h3 style='margin-bottom:12px;'></h3>
            <h4 style='color:#f4f4f5;'>AI Legal Reasoning</h4>
            <p style='color:#71717a; font-size:0.9rem;'>Generative AI fused with RAG provides clear, human-readable explanations and mitigation strategy.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='saas-card' style='text-align:center; height:100%;'>
            <h3 style='margin-bottom:12px;'></h3>
            <h4 style='color:#f4f4f5;'>Executive Reports</h4>
            <p style='color:#71717a; font-size:0.9rem;'>Export presentation-ready PDF analysis directly to your compliance officers and C-Suite.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        with st.expander("How This Works", expanded=True):
            st.write("This AI assistant analyzes your contract using machine learning and legal reasoning to detect risks and suggest improvements.")
            st.write("1. Upload a file on the left\n2. The system executes semantic segmentation\n3. The AI highlights any dangerous clauses for you.")
            
        st.markdown("<div style='text-align:center; margin-top:24px;'>", unsafe_allow_html=True)
        if st.button("Try Demo Contract", use_container_width=True, type="primary"):
            st.session_state["demo_to_load"] = "NDA"
            st.session_state["trigger_execution"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN VIEWS
# ══════════════════════════════════════════════════════════════════
def render_executive_summary(stats, risks):
    score = float(stats.get('risk_index', 0))
    status = "High Risk" if score >= 7 else ("Medium Risk" if score >= 4 else "Low Risk")
    
    st.markdown(f"""
    <div class="saas-card" style="border-top: 4px solid {'#f87171' if score>=7 else '#fbbf24' if score>=4 else '#34d399'};">
        <h4 style="margin-bottom: 24px; color:#f4f4f5;">Executive Summary</h4>
        <div style="display:flex; justify-content:space-between; margin-bottom: 24px;">
            <div>
                <div style="color:#71717a; font-size:0.8rem; text-transform:uppercase;">Contract Status</div>
                <div style="font-size:1.5rem; font-weight:600;">{status}</div>
            </div>
            <div>
                <div style="color:#71717a; font-size:0.8rem; text-transform:uppercase;">Overall Risk Score</div>
                <div style="font-size:1.5rem; font-family:'JetBrains Mono';">{score} / 10</div>
            </div>
            <div>
                <div style="color:#71717a; font-size:0.8rem; text-transform:uppercase;">Critical Clauses</div>
                <div style="font-size:1.5rem; font-family:'JetBrains Mono'; color:#f87171;">{stats.get('high_risk_clauses', 0)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_xai_block(item):
    xai_html = ""
    if item.get("xai_weights"):
        for word, weight in item.get("xai_weights").items():
            w_pct = min(weight * 100, 100)
            xai_html += f"""
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <div style="width:120px; color:#a1a1aa; font-size:0.8rem;">{word}</div>
                <div style="flex-grow:1; background:rgba(255,255,255,0.05); height:4px; border-radius:2px; margin:0 12px; position:relative;">
                    <div style="width:{w_pct}%; background:#f87171; height:100%; border-radius:2px; position:absolute; left:0;"></div>
                </div>
                <div style="font-size:0.8rem; color:#f87171; font-family:'JetBrains Mono';">+{weight:.2f}</div>
            </div>
            """
        xai_html += "<div style='margin-bottom:24px;'></div>"
    return xai_html


def tab_risk_analysis():
    st.session_state["current_step"] = 2
    render_stepper()
    
    state = st.session_state["agent_state_a"]
    stats = state["final_report"]["statistics"]
    risks = state["final_report"]["identified_risks"]

    render_executive_summary(stats, risks)

    if not risks:
        st.success("No high-risk clauses found. This contract appears mathematically safe.")
        st.info("Next Step → Generate full report (Export Tab)")
        return

    st.info("**Next Step →** Click on a clause below to understand the risk.")
    
    for idx, item in enumerate(risks):
        r_level = item.get("risk_level")
        chips = "".join([f"<span class='kw-chip'>{t}</span>" for t in item.get("linguistic_triggers", [])])
        
        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.3); padding:20px; border-radius:12px; border:1px solid rgba(255,255,255,0.05); border-left:4px solid {'#f87171' if r_level=='High Risk' else '#34d399'}; margin-bottom:12px;">
            <div style="display:flex; align-items:center; margin-bottom:12px;">
                <span class="badge {'badge-high' if r_level == 'High Risk' else 'badge-low'}">{r_level}</span>
                <span style="font-size:0.8rem; color:#71717a; margin-left:16px;">Confidence: {item['confidence']}</span>
                <div style="margin-left:auto;">{chips}</div>
            </div>
            <div style="font-size:0.95rem; color:#e4e4e7; line-height:1.6; margin-bottom:16px;">"{item['clause'][:250]}..."</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Explain Risk", key=f"btn_{idx}", help="Generate deep reasoning context"):
            st.session_state["selected_clause_idx"] = idx
            st.session_state["current_step"] = 3
            st.toast("Opening Explanations...", icon="None")

def tab_ai_assistant():
    if st.session_state["current_step"] < 3:
        st.session_state["current_step"] = 3
    render_stepper()
    
    state = st.session_state["agent_state_a"]
    risks = state["final_report"]["identified_risks"]
    idx = st.session_state.get("selected_clause_idx")
    
    if idx is None or idx >= len(risks):
        st.info("Please select 'Explain Risk' in the Risk Analysis tab.")
        return

    item = risks[idx]

    st.info("**Next Step →** Review insights here, then Generate full report on the Export Tab.")

    xai_html = render_xai_block(item)
    anomaly_html = ""
    if item.get("is_anomaly"):
        anomaly_html = f"<div style='background:rgba(192, 132, 252, 0.1); border-left:4px solid #c084fc; padding:12px; border-radius:8px; margin-bottom:24px; color:#e9d5ff;'><b style='color:#d8b4fe;'>Anomaly Detected:</b> This clause triggered Isolation Forest anomaly scores (Score: {item.get('anomaly_score')}). It represents an unusual zero-day formulation outside our training baseline.</div>"

    st.markdown(f"""
    <div class="saas-card" style="margin-bottom:0;">
        {anomaly_html}
        <h5 style="color:#f4f4f5; margin-bottom:8px;">Target Clause</h5>
        <div style="background:rgba(0,0,0,0.3); padding:16px; border-radius:12px; font-size:0.9rem; color:#a1a1aa; margin-bottom: 32px;">
            "{item['clause']}"
        </div>
        
        <h5 style="color:#f4f4f5; margin-bottom:16px;">1. Why is this risky? (ML Analysis)</h5>
        <div style="margin-bottom:12px; font-size:0.9rem; color:#a1a1aa;">The algorithm flagged these terms strongly correlating to liability traps:</div>
        {xai_html}
        
        <h5 style="color:#f4f4f5; margin-bottom:12px;">2. Legal Context (RAG Evidence)</h5>
        <p style="font-size:0.9rem; color:#c7d2fe; line-height:1.6; background:rgba(99, 102, 241, 0.05); padding:16px; border-radius:12px; margin-bottom:32px;">
            {item.get('legal_reference', '')}
        </p>
        
        <h5 style="color:#f4f4f5; margin-bottom:12px;">3. AI Reasoning & Reasoning</h5>
        <p style="font-size:0.95rem; color:#e4e4e7; line-height:1.6; border-left:2px solid #818cf8; padding-left:16px; margin-bottom:32px;">
            {item.get('explanation', '')}
        </p>
        
        <h5 style="color:#f4f4f5; margin-bottom:12px;">4. Suggested Fix</h5>
        <p style="font-size:0.95rem; color:#34d399; line-height:1.6; background:rgba(52, 211, 153, 0.05); padding:16px; border-radius:12px;">
            {item.get('mitigation', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)


def tab_analytics():
    st.session_state["current_step"] = 4
    render_stepper()
    state = st.session_state["agent_state_a"]
    results = state.get("ml_results", [])
    if not results: return

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(create_risk_distribution_chart(results), use_container_width=True)
    with col2: st.plotly_chart(create_confidence_trend_chart(results), use_container_width=True)


def tab_compare():
    st.session_state["current_step"] = 4
    render_stepper()
    
    state_a = st.session_state.get("agent_state_a")
    state_b = st.session_state.get("agent_state_b")

    if not state_a or not state_b:
        st.info("Upload Contract A and Contract B to run the delta comparison.")
        return

    try:
        score_a = state_a["final_report"]["statistics"]["risk_index"]
        score_b = state_b["final_report"]["statistics"]["risk_index"]
        comp = compare_contracts(
            st.session_state["file_name_a"], state_a["clauses"], state_a["ml_results"], score_a,
            st.session_state["file_name_b"], state_b["clauses"], state_b["ml_results"], score_b
        )
        
        st.markdown(f"""
        <div class="saas-card" style="margin-top:24px;">
            <div style="display:flex; justify-content:space-around; margin-bottom:24px; text-align:center;">
                <div><div style="color:#71717a; font-size:0.8rem;">Contract A Score</div><div style="font-size:1.5rem; font-weight:600;">{score_a}</div></div>
                <div><div style="color:#71717a; font-size:0.8rem;">Contract B Score</div><div style="font-size:1.5rem; font-weight:600;">{score_b}</div></div>
                <div><div style="color:#71717a; font-size:0.8rem;">Similarity</div><div style="font-size:1.5rem; font-weight:600; color:#818cf8;">{comp['semantic_alignment']}</div></div>
            </div>
            <p style="font-size:1rem; line-height:1.6; color:#a1a1aa; text-align:center;">{comp['summary']}</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Comparison computation failed: {e}")


def tab_export():
    st.session_state["current_step"] = 5
    render_stepper()
    state = st.session_state["agent_state_a"]
    
    st.success("Analysis Complete! Download your final artifacts below.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <h1></h1>
            <h4 style="color:#f4f4f5; margin-bottom:16px;">Executive Summary Report</h4>
            <p style="color:#71717a; font-size:0.9rem;">Presentation-ready PDF with risk scores and mitigations.</p>
        </div>""", unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(state.get("final_report", {}))
        st.download_button("📥 Download PDF", pdf_bytes, file_name="LexIQ_Risk_Report.pdf", use_container_width=True, type="primary")
        
    with c2:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <h1></h1>
            <h4 style="color:#f4f4f5; margin-bottom:16px;">Raw JSON Data</h4>
            <p style="color:#71717a; font-size:0.9rem;">Raw machine-learning extraction output for developers.</p>
        </div>""", unsafe_allow_html=True)
        json_b = report_to_json_string(state.get("final_report", {})).encode("utf-8")
        st.download_button("📥 Download JSON", json_b, file_name="LexIQ_Data.json", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
def main():
    init_session()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding-bottom: 24px;">
            <div style="font-size:3rem; line-height:1;"></div>
            <div style="font-size:1.5rem; font-weight:700; color:#f4f4f5;">LexIQ</div>
            <div style="font-size:0.8rem; color:#818cf8; letter-spacing:1px; margin-top:4px;">LEGAL INTELLIGENCE</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("Upload a contract to begin analysis")
        file_a = st.file_uploader("Upload Contract A", type=["pdf", "txt"], key="upload_a")
        file_b = st.file_uploader("Upload Contract B (Optional Comparison)", type=["pdf", "txt"], key="upload_b")
        
        st.markdown("<br/>", unsafe_allow_html=True)
        execute = st.button("Start Risk Analysis", use_container_width=True, type="primary")
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
                st.toast("Please upload a contract or select a Demo file.")
            
            if file_b:
                tb = get_text_from_file(file_b)
                if tb: process_document(tb, file_b.name, "b")

    # Render Base Content
    if not st.session_state["agent_state_a"]:
        render_onboarding()
    else:
        # Full App Nav
        t1, t2, t3, t4, t5 = st.tabs(["Risk Analysis", "AI Assistant", "Insights", "Compare", "Export"])
        
        with t1: tab_risk_analysis()
        with t2: tab_ai_assistant()
        with t3: tab_analytics()
        with t4: tab_compare()
        with t5: tab_export()

    # Footer
    st.markdown('<div class="footer">Trust & Transparency: This analysis is AI-generated and should be reviewed by a legal professional.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
