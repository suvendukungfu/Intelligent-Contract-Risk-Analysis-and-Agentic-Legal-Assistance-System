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
    create_risk_histogram,
    create_anomaly_scatter,
    create_feature_importance_chart,
    create_confusion_matrix,
    create_precision_recall_chart,
    create_data_drift_chart,
    create_llm_reliability_chart,
    create_rag_observability_chart
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
    .step:not(:last-child)::after { content: '>'; position: absolute; right: -10px; color: #3f3f46; font-size: 0.9rem; }
    
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
        <div class="step {cls(2)}">2. Analyze</div>
        <div class="step {cls(3)}">3. Risks</div>
        <div class="step {cls(4)}">4. Explain</div>
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
    st.markdown("<h3 style='text-align:center; color:#818cf8; font-weight:400; font-family:Outfit; margin-bottom:8px;'>AI Contract Risk Intelligence Platform</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#a1a1aa; font-size:1.1rem; margin-bottom:48px;'>Upload contracts. Detect risks. Get legal insights.</p>", unsafe_allow_html=True)
    
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
    
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        if st.button("Upload Contract", use_container_width=True):
            st.info("Please use the sidebar on the left to upload your document.")
    with bc2:
        if st.button("Try Demo NDA", type="primary", use_container_width=True):
            st.session_state["demo_to_load"] = "NDA"
            st.session_state["trigger_execution"] = True
            st.rerun()
    with bc3:
        if st.button("Learn How It Works", use_container_width=True):
            st.session_state["show_guide"] = True

    if st.session_state.get("show_guide"):
        with st.expander("The LexIQ Methodology", expanded=True):
            st.markdown("""
            **1. Semantic Fragmentation:** We use NLP to break your contract into distinct legal clauses.
            **2. ML Risk Scoring:** Every clause is run through a Logistic Regression model trained on 10,000+ legal precedents.
            **3. XAI (Explainable AI):** The system extracts word-level weights to show exactly *why* it flagged a clause.
            **4. Agentic RAG:** A Mistral-based LLM checks your clause against a verified legal knowledge base to suggest fixes.
            """)
            if st.button("Close Guide"): 
                st.session_state["show_guide"] = False
                st.rerun()


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


def tab_risk_analysis():
    st.session_state["current_step"] = 2
    render_stepper()
    
    state = st.session_state["agent_state_a"]
    stats = state["final_report"]["statistics"]
    risks = state["final_report"]["identified_risks"]

    render_executive_summary(stats, risks)

    if not risks:
        st.success("No high-risk clauses found. This contract appears mathematically safe.")
        st.info("Next Step - Generate full report (Export Tab)")
        return

    st.info("Next Step - Click on a clause below to understand the risk.")
    
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
        
        if st.session_state.get("selected_clause_idx") != idx:
            if st.button("Explain Risk", key=f"btn_{idx}", help="Generate deep reasoning context"):
                st.session_state["selected_clause_idx"] = idx
                st.session_state["current_step"] = 3
                st.rerun()
        else:
            if st.button("Collapse Explanation", key=f"btn_close_{idx}"):
                st.session_state["selected_clause_idx"] = None
                st.rerun()
                
            # Restore the XAI (Risk Word) graph logic natively
            def render_linguistic_importance(xai_weights):
                if not xai_weights: return
                st.markdown("**Linguistic Trigger Importance:**")
                for word, weight in xai_weights.items():
                    w_pct = min(weight * 100, 100)
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:4px;">
                        <div style="width:100px; color:#a1a1aa; font-size:0.75rem;">{word}</div>
                        <div style="flex-grow:1; background:rgba(255,255,255,0.05); height:4px; border-radius:2px; margin:0 12px; position:relative;">
                            <div style="width:{w_pct}%; background:#f87171; height:100%; border-radius:2px; position:absolute; left:0;"></div>
                        </div>
                        <div style="font-size:0.75rem; color:#f87171; font-family:'JetBrains Mono';">+{weight:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

            with st.container():
                st.markdown("---")
                
                if item.get("is_anomaly"):
                    st.warning("Anomaly Detected: This clause represents an unusual legal formulation outside standard training baselines.")

                # Redesigned Explanation UI
                st.markdown(f"### [ {r_level} ] Risk Assessment")
                
                cl1, cl2 = st.columns([2, 1])
                with cl1:
                    st.markdown("**AI Legal Interpretation (Plain English):**")
                    st.write(item.get('explanation', 'Awaiting deep reasoning...'))
                with cl2:
                    st.markdown("**Model Intelligence:**")
                    conf_val = item.get('confidence', 0)
                    st.metric("Confidence Score", f"{conf_val * 100:.1f}%" if isinstance(conf_val, float) else "N/A")

                st.markdown("**Machine Learning Triggers:**")
                if item.get("xai_weights"):
                    render_linguistic_importance(item.get("xai_weights"))
                else:
                    st.write("Keyword triggers extracted from semantic vector space.")

                st.markdown("**Suggested Mitigation Strategy:**")
                st.info(item.get('mitigation', 'Consider standardizing this clause with a liability cap.'))
                
                st.markdown("<br/>", unsafe_allow_html=True)
                if st.button("Next: View Comparative Insights", use_container_width=True):
                    st.session_state["current_step"] = 4
                    st.rerun()


def tab_ai_assistant():
    st.info("The AI Assistant explanations are now seamlessly integrated directly beneath the clauses in the Risk Analysis tab! Go back and click 'Explain Risk' to see them instantly.")


def tab_analytics():
    st.session_state["current_step"] = 4
    render_stepper()
    state = st.session_state["agent_state_a"]
    results = state.get("ml_results", [])
    if not results:
        st.info("No analytics data available yet.")
        return

    report_stats = state.get("final_report", {}).get("statistics", {})
    avg_conf = float(str(report_stats.get('avg_confidence', '0%')).replace('%', ''))
    anomaly_count = sum(1 for r in results if r.get('is_anomaly'))
    explanations = state.get("explanations", [])
    retrieval_data = state.get("retrieved_contexts", [])
    
    st.markdown("### System Health and Overview (Executive Panel)")
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.markdown(f"<div class='saas-card'><h4>Overall Risk Index</h4><div style='font-size:2rem; font-weight:700; color:#818cf8;'>{report_stats.get('risk_index', 0)}/10</div><div style='color:#a1a1aa; font-size:0.8rem;'>System risk classification</div></div>", unsafe_allow_html=True)
    with kc2:
        st.markdown(f"<div class='saas-card'><h4>Average Confidence</h4><div style='font-size:2rem; font-weight:700; color:#34d399;'>{avg_conf}%</div><div style='color:#a1a1aa; font-size:0.8rem;'>AI inference certainty</div></div>", unsafe_allow_html=True)
    with kc3:
        st.markdown(f"<div class='saas-card'><h4>Zero-Day Anomalies</h4><div style='font-size:2rem; font-weight:700; color:#c084fc;'>{anomaly_count}</div><div style='color:#a1a1aa; font-size:0.8rem;'>Outlier structures found</div></div>", unsafe_allow_html=True)
    with kc4:
        st.markdown(f"<div class='saas-card'><h4>Context Pulled</h4><div style='font-size:2rem; font-weight:700; color:#fbbf24;'>{sum(len(r.get('context_chunks', [])) for r in retrieval_data)}</div><div style='color:#a1a1aa; font-size:0.8rem;'>ChromaDB vectors used</div></div>", unsafe_allow_html=True)

    # Automated Alert System
    if avg_conf < 85.0:
        st.warning("Warning: Low Average Confidence. The ML Model is exhibiting low certainty for this document structure. Proceed with manual legal review.")
    if anomaly_count > 0:
        st.error(f"Alert: {anomaly_count} Anomalies Detected. The Isolation Forest model flagged clauses severely disjointed from standard baseline patterns.")
    if abs(25 - (sum(len(str(r["clause"]).split()) for r in results) / max(len(results), 1))) > 20: 
        st.info("Data Drift Notice: The average clause length significantly deviates from the historical training distributions, potentially lowering accuracy.")

    st.markdown("---")
    
    # Filters
    st.markdown("### Interactive Filters")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        selected_risks = st.multiselect("Filter by Risk Level", ["Low Risk", "Unknown", "High Risk"], default=["Low Risk", "Unknown", "High Risk"])
    with fcol2:
        conf_min = st.slider("Minimum AI Confidence (%)", 0, 100, 0)
        
    filtered_results = [r for r in results if r["risk_level"] in selected_risks and (r["confidence"] * 100) >= conf_min]
    
    st.markdown("---")
    st.markdown("### 1. Model Evaluation Metrics (Baseline)")
    st.info("Demonstrates the machine learning agent's standardized validation metrics from training.")
    mc1, mc2 = st.columns(2)
    with mc1: 
        st.plotly_chart(create_confusion_matrix(), use_container_width=True)
        st.caption("Confusion Matrix: Standardized evaluation showing True Positives vs False Positives.")
    with mc2: 
        st.plotly_chart(create_precision_recall_chart(), use_container_width=True)
        st.caption("Precision: How accurate the risk predictions are. Recall: How many risky clauses are correctly identified.")

    st.markdown("---")
    st.markdown("### 2. Data Drift & Clause Analytics")
    c1, c2 = st.columns(2)
    with c1: 
        st.plotly_chart(create_risk_distribution_chart(filtered_results), use_container_width=True)
        st.caption("Risk Categories: Ratio of High Risk to Low Risk clauses in your document.")
    with c2: 
        st.plotly_chart(create_data_drift_chart(filtered_results), use_container_width=True)
        st.caption("Data Drift: Compares current clause lengths against static baseline boundaries. Shifted geometry implies model instability.")

    st.markdown("---")
    st.markdown("### 3. Real-time Confidence & Flow Analytics")
    st.plotly_chart(create_confidence_trend_chart(filtered_results), use_container_width=True)
    st.caption("Prediction Confidence Monitoring: Shows how confident the AI is across the chronological flow of the document. Spikes indicate clarity; dips indicate confusing terminology.")

    st.markdown("---")
    st.markdown("### 4. Observability: RAG & LLM Extraction Health")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.plotly_chart(create_rag_observability_chart(retrieval_data), use_container_width=True)
        st.caption("RAG Tracking: Tracks the raw injection vectors passed to the LLM. Empty plots mean semantic search yielded nothing.")
    with rc2:
        st.plotly_chart(create_llm_reliability_chart(explanations), use_container_width=True)
        st.caption("LLM Safety: Maps response token-length constraints to detect uncontrolled hallucinatory generations (long lengths) or truncation errors (short lengths).")

    st.markdown("---")
    st.markdown("### 5. Explainability (XAI) & Anomaly Tracking")
    x1, x2 = st.columns(2)
    with x1: 
        st.plotly_chart(create_feature_importance_chart(filtered_results), use_container_width=True)
        st.caption("Feature Importance Tracking: TF-IDF keywords that act as strong liability traps.")
    with x2: 
        st.plotly_chart(create_anomaly_scatter(filtered_results), use_container_width=True)
        st.caption("Anomaly Detection Monitoring: Plots normal clauses vs outliers to detect previously unseen or 'Zero-Day' legal formulations.")


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
        score_b = state_b["final_report_b"]["statistics"]["risk_index"] if "final_report_b" in state_b else state_b["final_report"]["statistics"]["risk_index"]
        
        comp = compare_contracts(
            st.session_state["file_name_a"], state_a["clauses"], state_a["ml_results"], score_a,
            st.session_state["file_name_b"], state_b["clauses"], state_b["ml_results"], score_b
        )
        
        # 1. Executive Summary Cards
        st.markdown("### Executive Risk Comparison")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Risk Index (A vs B)", f"{score_a}", f"{score_a - score_b:+.1f}", delta_color="inverse")
        with c2:
            st.metric("Similarity", comp['semantic_alignment'])
        with c3:
            st.metric("Risk Delta", f"{comp['high_risk_diff']} Clauses", delta_color="off")

        st.markdown(f"""
        <div class="saas-card" style="background: rgba(129, 140, 248, 0.05); border: 1px solid rgba(129, 140, 248, 0.2);">
            <h4 style="color:#818cf8; margin-bottom:8px;">AI Verdict & Selection Recommendation</h4>
            <p style="color:#e4e4e7; font-size:0.95rem; line-height:1.6;">{comp['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

        # 3. Missing Protections Side-by-Side
        st.markdown("---")
        st.markdown("### Structural Gaps & Missing Protections")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"**Missing in {st.session_state['file_name_a']}**")
            if comp['missing_in_a']:
                for item in comp['missing_in_a']:
                    st.markdown(f"<div style='color:#f87171; border-left:2px solid #f87171; padding-left:10px; margin-bottom:8px; font-size:0.85rem;'>[!] Missing {item}</div>", unsafe_allow_html=True)
            else:
                st.success("All standard protections present.")
        with m2:
            st.markdown(f"**Missing in {st.session_state['file_name_b']}**")
            if comp['missing_in_b']:
                for item in comp['missing_in_b']:
                    st.markdown(f"<div style='color:#f87171; border-left:2px solid #f87171; padding-left:10px; margin-bottom:8px; font-size:0.85rem;'>[!] Missing {item}</div>", unsafe_allow_html=True)
            else:
                st.success("All standard protections present.")

        # 4. Detailed Delta Comparison
        if comp['mapped_diffs']:
            st.markdown("---")
            st.markdown("### Clause-Level Semantic Deltas")
            for diff in comp['mapped_diffs']:
                with st.expander(f"Analysis: {diff['topic']}"):
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown(f"**Contract A (Risky Formulation)**")
                        st.caption(f"\"{diff['clause_a']}...\"")
                    with dc2:
                        st.markdown(f"**Contract B (Balanced Alternative)**")
                        st.caption(f"\"{diff['clause_b']}...\"")
                    st.markdown(f"**System Insight:** {diff['insight']}")
        
        st.markdown("<br/>", unsafe_allow_html=True)
        import json
        comp_json = json.dumps(comp, indent=4).encode('utf-8')
        st.download_button("Export Comparison Data (JSON)", comp_json, "Comparison_Report.json", use_container_width=True)

    except Exception as e:
        st.error(f"Comparison computation failed: {e}")
        st.exception(e)


def tab_export():
    st.session_state["current_step"] = 5
    render_stepper()
    state = st.session_state["agent_state_a"]
    
    st.success("Analysis Complete! Review your final report or download the artifacts below.")
    
    report = state.get("final_report", {})
    if report:
        st.markdown("""
        <div class="saas-card" style="background: rgba(0,0,0,0.5); border-top: 4px solid #818cf8;">
            <h3 style="color:#f4f4f5; font-family:'Outfit'; margin-bottom:8px;">Final Intelligence Report</h3>
            <p style="color:#a1a1aa; font-family:'JetBrains Mono'; font-size:0.85rem; margin-bottom:32px;">Generated by LexIQ Enterprise Legal AI</p>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Overall Risk Index:** `{report.get('statistics', {}).get('risk_index', 0)} / 10`")
        st.markdown(f"**Total High-Risk Clauses:** `{report.get('statistics', {}).get('high_risk_clauses', 0)}`")
        
        for idx, risk in enumerate(report.get("identified_risks", [])):
            clean_clause = str(risk.get('clause', '')).strip()
            
            st.markdown(f"""
            <div style="margin-top:24px; padding-top:24px; border-top:1px solid rgba(255,255,255,0.05);">
                <div style="color:#818cf8; font-weight:700; margin-bottom:8px; font-family:'Outfit';">Clause {idx+1} [ {risk.get('risk_level', 'Unknown')} ]</div>
                <div style="font-size:0.9rem; color:#a1a1aa; font-style:italic; border-left:2px solid #3f3f3f; padding-left:12px; margin-bottom:16px;">
                    "{clean_clause}"
                </div>
            """, unsafe_allow_html=True)
            
            if risk.get("risk_level") == "Low Risk":
                 st.markdown("<p style='color:#34d399; font-size:0.9rem;'>Verification passed. Standard language detected.</p>", unsafe_allow_html=True)
            else:
                 st.markdown(f"**Triggers Identified:** `{', '.join(risk.get('linguistic_triggers', []))}`")
                 st.markdown(f"**AI Reasoning:** {risk.get('explanation', 'N/A')}")
                 st.markdown(f"<span style='color:#fca5a5;'>**Suggested Mitigation:** {risk.get('mitigation', 'N/A')}</span>", unsafe_allow_html=True)
                 
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <h1></h1>
            <h4 style="color:#f4f4f5; margin-bottom:16px;">Executive Summary Report</h4>
            <p style="color:#71717a; font-size:0.9rem;">Presentation-ready PDF with risk scores and mitigations.</p>
        </div>""", unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(state.get("final_report", {}))
        st.download_button("Download PDF", pdf_bytes, file_name="LexIQ_Risk_Report.pdf", use_container_width=True, type="primary")
        
    with c2:
        st.markdown("""<div class="saas-card" style="text-align:center;">
            <h1></h1>
            <h4 style="color:#f4f4f5; margin-bottom:16px;">Raw JSON Data</h4>
            <p style="color:#71717a; font-size:0.9rem;">Raw machine-learning extraction output for developers.</p>
        </div>""", unsafe_allow_html=True)
        json_b = report_to_json_string(state.get("final_report", {})).encode("utf-8")
        st.download_button("Download JSON", json_b, file_name="LexIQ_Data.json", use_container_width=True)


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
        
        with st.expander("Step-by-Step Guide", expanded=True):
            st.markdown("""
            **1. Upload:** Add your PDF/TXT contract.
            **2. Analyze:** The AI segments and scores risk.
            **3. Risks:** View identified liability traps.
            **4. Explain:** Understand the legal 'why'.
            **5. Report:** Download your PDF audit.
            """)

        st.info("Professional AI Audit Tool")
        file_a = st.file_uploader("Upload Primary Contract", type=["pdf", "txt"], key="upload_a", help="The document you wish to analyze for risks.")
        file_b = st.file_uploader("Comparison Contract (Optional)", type=["pdf", "txt"], key="upload_b", help="Upload a secondary doc to find semantic deltas.")
        
        st.markdown("<br/>", unsafe_allow_html=True)
        execute = st.button("Start AI Analysis", use_container_width=True, type="primary", help="Trigger semantic segmentation and ML risk inference.")
        if execute: st.session_state["trigger_execution"] = True

    # Handle Pending Executions
    if st.session_state["trigger_execution"]:
        st.session_state["trigger_execution"] = False
        
        if st.session_state["demo_to_load"]:
            t, n = load_demo_contract(st.session_state["demo_to_load"])
            if t: 
                st.toast("Welcome to the LexIQ Guided Tour!", icon=None)
                st.session_state["is_demo_mode"] = True
            st.session_state["demo_to_load"] = None
        else:
            if file_a: 
                t = get_text_from_file(file_a)
                if t: process_document(t, file_a.name, "a")
            else: 
                st.toast("Notice: Please upload a contract or select a Demo file.")
            
            if file_b:
                tb = get_text_from_file(file_b)
                if tb: process_document(tb, file_b.name, "b")

    # Demo Tour Highlighting
    if st.session_state.get("is_demo_mode"):
        step = st.session_state["current_step"]
        if step == 2:
            st.toast("Guide: Analyzing risks and clustering anomalies...", icon=None)
        elif step == 3:
            st.toast("Guide: Identified liability traps. Click 'Explain' for legal analysis.", icon=None)
        elif step == 4:
            st.toast("Guide: Comparative monitoring and confidence metrics.", icon=None)
        elif step == 5:
            st.toast("Guide: Final professional audit report generation.", icon=None)
            st.session_state["is_demo_mode"] = False # End tour at report

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
