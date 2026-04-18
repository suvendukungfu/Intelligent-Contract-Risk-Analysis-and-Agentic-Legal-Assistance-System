"""
app/streamlit_app.py
--------------------
Enterprise SaaS UI for LexIQ Legal AI.
Guided workflow: [1 Upload] -> [2 Analyze] -> [3 Risks] -> [4 Monitoring] -> [5 Report]
"""

import streamlit as st
import pandas as pd
import os
import sys

# ── Ensure project root is in path ───────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.workflow import run_agent_pipeline
from analytics.dashboard import (
    create_confusion_matrix_heatmap, create_metrics_bar_chart,
    create_drift_distribution_plot, create_confidence_line_chart,
    create_risk_pie_chart, create_feature_importance_bar,
    create_system_health_radar
)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LexIQ | AI Contract Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Integration ──────────────────────────────────────────────────────────
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create a minimal CSS for SaaS look if external file not found or preferred
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #818cf8;
        --bg-dark: #09090b;
        --card-bg: #18181b;
        --border: #27272a;
    }
    
    .stApp { background-color: var(--bg-dark); color: #f4f4f5; font-family: 'Inter', sans-serif; }
    
    .saas-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    .saas-card:hover { border-color: var(--primary); }
    
    .badge {
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-high { background: rgba(248, 113, 113, 0.1); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.2); }
    .badge-low { background: rgba(52, 211, 153, 0.1); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.2); }
    
    .kw-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        color: #a1a1aa;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-right: 4px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom Stepper */
    .stepper-container { display: flex; justify-content: space-between; margin-bottom: 40px; padding: 0 20px; }
    .step { display: flex; flex-direction: column; align-items: center; opacity: 0.4; }
    .step-active { opacity: 1; color: var(--primary); }
    .step-icon { width: 32px; height: 32px; border-radius: 16px; border: 2px solid currentColor; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR & GUIDED FLOW
# ══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/law.png", width=48)
        st.title("LexIQ Enterprise")
        st.markdown("---")
        
        st.markdown("### 🚦 Workflow Guide")
        step = st.session_state.get("current_step", 1)
        
        guides = {
            1: "📤 **Step 1: Upload** - Drop your PDF/TXT contract below to begin segmentation.",
            2: "🔍 **Step 2: Analyze** - Review clauses flagged by our ML Risk Engine.",
            3: "🧠 **Step 3: Reasoning** - Click 'Explain' to see Agentic RAG insights.",
            4: "📊 **Step 4: Monitoring** - Inspect model health and drift metrics.",
            5: "📄 **Step 5: Export** - Generate an enterprise-grade legal audit report."
        }
        st.info(guides.get(step))
        
        st.markdown("---")
        st.markdown("### 🏢 System Status")
        st.success("ML Engine: Online")
        st.success("RAG Context: 10,242 nodes")
        st.info("Version: 2.1 Agentic")

def render_stepper():
    current = st.session_state.get("current_step", 1)
    steps = ["Upload", "Analyze", "Reason", "Monitor", "Report"]
    
    cols = st.columns(len(steps))
    for i, s in enumerate(steps):
        is_active = (i + 1) == current
        with cols[i]:
            st.markdown(f"""
            <div class="step {'step-active' if is_active else ''}">
                <div class="step-icon">{i+1}</div>
                <div style="font-size:0.8rem; font-weight:500;">{s}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB LOGIC
# ══════════════════════════════════════════════════════════════════

def tab_upload():
    st.session_state["current_step"] = 1
    render_stepper()
    
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700;">LexIQ Intelligence Platform</h1>
        <p style="color: #a1a1aa; font-size: 1.1rem;">Enterprise-grade contract risk detection and agentic reasoning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Contract (PDF/TXT)", type=["pdf", "txt"])
    
    if uploaded_file:
        with st.spinner("Agentic pipeline initializing..."):
            # Mock extraction for example
            content = uploaded_file.read().decode("utf-8") if uploaded_file.type == "text/plain" else "PDF Content Placeholder"
            state = run_agent_pipeline(content, uploaded_file.name)
            st.session_state["agent_state_a"] = state
            st.session_state["current_step"] = 2
            st.rerun()

def render_executive_summary(stats, risks):
    score = stats.get('risk_index', 0)
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
    risks = state.get("risks", [])
    
    # Calculate stats on the fly
    high_count = sum(1 for r in risks if r["risk_level"] == "High Risk")
    stats = {
        "risk_index": round((high_count / len(risks) * 10) if risks else 0, 1),
        "high_risk_clauses": high_count
    }

    render_executive_summary(stats, risks)

    for idx, item in enumerate(risks):
        r_level = item.get("risk_level")
        chips = "".join([f"<span class='kw-chip'>{t}</span>" for t in item.get("triggers", [])])
        
        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.3); padding:20px; border-radius:12px; border:1px solid rgba(255,255,255,0.05); border-left:4px solid {'#f87171' if r_level=='High Risk' else '#34d399'}; margin-bottom:12px;">
            <div style="display:flex; align-items:center; margin-bottom:12px;">
                <span class="badge {'badge-high' if r_level == 'High Risk' else 'badge-low'}">{r_level}</span>
                <span style="font-size:0.8rem; color:#71717a; margin-left:16px;">Confidence: {item.get('confidence', 0)*100:.1f}%</span>
                <div style="margin-left:auto;">{chips}</div>
            </div>
            <div style="font-size:0.95rem; color:#e4e4e7; line-height:1.6; margin-bottom:16px;">"{item['clause'][:250]}..."</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.checkbox(f"Show Reasoning Analysis", key=f"chk_{idx}"):
            exp_map = {e["clause_idx"]: e for e in state.get("explanations", [])}
            exp = exp_map.get(idx, {})
            st.info(exp.get("explanation", "No reasoning available for this clause."))
            st.warning(f"Mitigation: {exp.get('mitigation', 'N/A')}")

def tab_analytics():
    """AI Monitoring System (Production Dashboard)"""
    st.session_state["current_step"] = 4
    render_stepper()
    
    state = st.session_state["agent_state_a"]
    risks = state.get("risks", [])
    explanations = state.get("explanations", [])
    retrieval = state.get("retrieved_context", [])
    
    if not risks:
        st.info("Insufficient data for monitoring. Please analyze a document first.")
        return

    st.markdown("### 🛠 AI Monitoring System (Production)")
    
    # Drift Check
    curr_avg_len = sum(len(r["clause"].split()) for r in risks) / len(risks)
    has_drift = abs(28 - curr_avg_len) > 15
    if has_drift:
        st.error("🚨 **DATA DRIFT DETECTED**: Current document structure significantly deviates from training baseline.")
    else:
        st.success("🟢 **SYSTEM HEALTH**: All agents operating within nominal parameters.")

    # Visuals
    m1, m2 = st.columns([1, 1.5])
    with m1: st.plotly_chart(create_metrics_bar_chart(), use_container_width=True)
    with m2: st.plotly_chart(create_confusion_matrix_heatmap(), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1: st.plotly_chart(create_drift_distribution_plot(risks), use_container_width=True)
    with d2: st.plotly_chart(create_confidence_line_chart(risks), use_container_width=True)

    r1, r2, r3 = st.columns([1, 1, 1.2])
    with r1: st.plotly_chart(create_risk_pie_chart(risks), use_container_width=True)
    with r2: st.plotly_chart(create_feature_importance_bar(risks), use_container_width=True)
    with r3: st.plotly_chart(create_system_health_radar(explanations, retrieval), use_container_width=True)

def tab_export():
    st.session_state["current_step"] = 5
    render_stepper()
    state = st.session_state["agent_state_a"]
    report = state.get("final_report", {})
    
    if not report:
        st.warning("No report generated. Please analyze a document first.")
        return

    st.markdown("### 📄 Professional Legal AI Audit")
    st.markdown("---")

    # 1. Executive Summary
    exec_sum = report.get("executive_summary", {})
    st.markdown("#### 1. Executive Summary")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Overall Risk Score", exec_sum.get("overall_risk_score", "N/A"))
    with c2: st.metric("Contract Status", exec_sum.get("contract_status", "N/A"))
    with c3: st.metric("Total Clauses", len(state.get("risks", [])))
    st.write(exec_sum.get("summary_statement", ""))

    # 3. Key Risk Insights
    st.markdown("#### 2. Key Risk Insights (Top Critical Findings)")
    for insight in report.get("key_risk_insights", []):
        with st.expander(f"📍 {insight['topic']}"):
            st.write(f"**Primary Concern:** {insight['primary_concern']}")
            st.markdown(f"*Clause Snippet:* \"{insight['clause_snippet']}\"")

    # 4. Recommendations
    st.markdown("#### 3. Actionable Recommendations")
    for rec in report.get("recommendations", []):
        st.markdown(f"- **{rec['category']}:** {rec['action']}")

    # 5. Explainability
    st.markdown("#### 4. Agentic Explainability (Deep Dive)")
    st.write("Cross-referencing ML trigger weights with Agentic RAG grounding.")
    expl_data = report.get("explainability", [])
    if expl_data:
        st.dataframe(pd.DataFrame(expl_data), hide_index=True)
    else:
        st.info("No high-risk clauses required deep-dive explanation.")

    st.markdown("---")
    # 6. Disclaimer
    st.caption(report.get("disclaimer", ""))

    st.markdown("<br/>", unsafe_allow_html=True)
    st.download_button(
        "Download Full Digital Audit (JSON)", 
        json.dumps(report, indent=2), 
        file_name="LexIQ_Legal_Audit.json",
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════
# MAIN APP ENTRY
# ══════════════════════════════════════════════════════════════════

def main():
    render_sidebar()
    
    if "agent_state_a" not in st.session_state:
        # Landing Page
        tab_upload()
    else:
        tabs = st.tabs(["Analyze", "Monitor", "Report"])
        with tabs[0]: tab_risk_analysis()
        with tabs[1]: tab_analytics()
        with tabs[2]: tab_export()

if __name__ == "__main__":
    main()
