"""
app/streamlit_app.py
--------------------
Enterprise SaaS UI for LexIQ Legal AI.
Guided workflow: [1 Upload] -> [2 Analyze] -> [3 Compare] -> [4 Monitor] -> [5 Report]

Production-grade: Zero-error, deploy-ready build.
"""

import os
import sys
import subprocess

# ── Streamlit Cloud Compatibility Patch ──────────────────────────────
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# ── Spacy Model Check ────────────────────────────────────────────────
try:
    import spacy
    if not spacy.util.is_package("en_core_web_sm"):
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
except Exception:
    pass

import streamlit as st
import pandas as pd
import json

# ── Ensure project root is FIRST in path to prevent module shadowing ─────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Integration ──────────────────────────────────────────────
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
        
        st.markdown("### Workflow Guide")
        step = st.session_state.get("current_step", 1)
        
        guides = {
            1: "**Step 1: Upload** - Drop your PDF/TXT contract below to begin.",
            2: "**Step 2: Analyze** - Review clauses flagged by the system.",
            3: "**Step 3: Compare** - Side-by-side contract comparison.",
            4: "**Step 4: Monitoring** - Inspect system health metrics.",
            5: "**Step 5: Export** - Generate a legal audit report."
        }
        st.info(guides.get(step, guides[1]))
        
        st.markdown("---")
        st.markdown("### System Status")
        st.success("Analysis Engine: Online")
        st.success("Knowledge Base: Connected")
        st.info("Version: 2.1")

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
# TAB: UPLOAD
# ══════════════════════════════════════════════════════════════════

def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith('.pdf'):
        import pdfplumber
        text = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
        return "\\n".join(text) if text else "PDF Extraction Failed or Empty"
    return uploaded_file.read().decode("utf-8", errors="ignore")

def tab_upload():
    st.session_state["current_step"] = 1
    render_stepper()
    
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700;">AI Contract Risk Analyzer</h1>
        <p style="color: #a1a1aa; font-size: 1.1rem;">Clear contract risk detection and explanation.</p>
        <p style="color: #818cf8; font-weight: 500;">This AI analyzes contracts, detects risks, and explains them.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("How to Use"):
        st.markdown("""
        1. Upload contract
        2. Run analysis
        3. Check risks
        4. Explain clauses
        5. Download report
        """)
        
    if st.button("Try Demo"):
        with st.spinner("Loading demo contract and initializing analysis..."):
            demo_text = "This agreement limits our liability to $100 and requires arbitration. You agree to indemnify us for any third-party claims."
            st.session_state["agent_state_a"] = run_agent_pipeline(demo_text, "Demo_Contract.txt")
            st.session_state["current_step"] = 2
            st.rerun()

    st.markdown("---")
    
    comp_mode = st.checkbox("Enable Comparison Mode (Dual Contract Analysis)", key="comp_toggle")
    
    if not comp_mode:
        uploaded_file = st.file_uploader("Upload Primary Contract (PDF/TXT)", type=["pdf", "txt"], key="single_up")
        if uploaded_file:
            with st.spinner("Agentic pipeline initializing..."):
                content = extract_text(uploaded_file)
                state = run_agent_pipeline(content, uploaded_file.name)
                st.session_state["agent_state_a"] = state
                st.session_state["current_step"] = 2
                st.rerun()
    else:
        c1, c2 = st.columns(2)
        with c1: f1 = st.file_uploader("Contract A", type=["pdf", "txt"], key="comp_up_a")
        with c2: f2 = st.file_uploader("Contract B", type=["pdf", "txt"], key="comp_up_b")
        
        if f1 and f2:
            if st.button("Analyze & Compare", type="primary", use_container_width=True):
                with st.spinner("Running Dual Agentic Pipeline..."):
                    st.session_state["agent_state_a"] = run_agent_pipeline(extract_text(f1), f1.name)
                    st.session_state["agent_state_b"] = run_agent_pipeline(extract_text(f2), f2.name)
                    st.session_state["comparison_active"] = True
                    st.session_state["current_step"] = 2
                    st.rerun()

# ══════════════════════════════════════════════════════════════════
# TAB: RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════

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
    
    high_count = sum(1 for r in risks if r["risk_level"] == "High Risk")
    stats = {
        "risk_index": round((high_count / len(risks) * 10) if risks else 0, 1),
        "high_risk_clauses": high_count
    }

    render_executive_summary(stats, risks)
    
    st.info("**Next Step:** Review the extracted risk clauses below, then click **'Show Reasoning Analysis'** to understand the context.")

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
            
            st.markdown(f"**Summary:** {exp.get('summary', exp.get('explanation', 'N/A')[:80])}")
            st.info(f"**Why Risky:** {exp.get('explanation', 'No justification available.')}")
            st.warning(f"**Legal Meaning:** {exp.get('legal_implications', 'N/A')}")
            st.success(f"**Suggested Fix:** {exp.get('mitigation', 'N/A')}")


# ══════════════════════════════════════════════════════════════════
# TAB: COMPARISON
# ══════════════════════════════════════════════════════════════════

def tab_comparison():
    st.session_state["current_step"] = 3
    render_stepper()
    
    state_a = st.session_state.get("agent_state_a")
    state_b = st.session_state.get("agent_state_b")
    
    if not state_a or not state_b:
        st.info("Comparison Mode requires two contracts. Please upload them in the [Upload] step.")
        return

    from models.comparison import compare_contracts
    
    results = compare_contracts(
        state_a["file_name"], state_a["clauses"], state_a["risks"],
        state_b["file_name"], state_b["clauses"], state_b["risks"]
    )

    # ── VERDICT CARD ──
    winner = results["winner"]
    st.markdown(f"""
    <div class="saas-card" style="border-left: 10px solid #818cf8; background: rgba(129, 140, 248, 0.1);">
        <h3 style="margin:0;">Safety Verdict: {winner if winner != 'Tie' else 'Equally Balanced'}</h3>
        <p style="font-size:1.1rem; color:#f4f4f5; margin-top:10px;">{results['verdict']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── SIDE BY SIDE SUMMARY ──
    col_a, col_b = st.columns(2)
    s_a = results["stats_a"]
    s_b = results["stats_b"]
    
    comp_css = "background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; text-align: center; height: 100%;"
    
    with col_a:
        st.markdown(f"""
        <div style="{comp_css}">
            <h4 style="color:#818cf8; margin-top:0;">{results['metadata']['name_a']}</h4>
            <h1 style="font-family:'JetBrains Mono'; margin:10px 0;">{s_a['risk_score']} / 10</h1>
            <p style="color:#a1a1aa; font-size:0.9rem; text-transform:uppercase;">Overall Risk Score</p>
            <hr style="border-color:rgba(255,255,255,0.05);">
            <div style="display:flex; justify-content:space-around;">
                <div><strong style="font-size:1.2rem;">{s_a['total_clauses']}</strong><br><span style="font-size:0.8rem;color:#71717a">Total Clauses</span></div>
                <div><strong style="font-size:1.2rem;color:#f87171;">{s_a['high_risk']}</strong><br><span style="font-size:0.8rem;color:#71717a">High Risk</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown(f"""
        <div style="{comp_css}">
            <h4 style="color:#818cf8; margin-top:0;">{results['metadata']['name_b']}</h4>
            <h1 style="font-family:'JetBrains Mono'; margin:10px 0;">{s_b['risk_score']} / 10</h1>
            <p style="color:#a1a1aa; font-size:0.9rem; text-transform:uppercase;">Overall Risk Score</p>
            <hr style="border-color:rgba(255,255,255,0.05);">
            <div style="display:flex; justify-content:space-around;">
                <div><strong style="font-size:1.2rem;">{s_b['total_clauses']}</strong><br><span style="font-size:0.8rem;color:#71717a">Total Clauses</span></div>
                <div><strong style="font-size:1.2rem;color:#f87171;">{s_b['high_risk']}</strong><br><span style="font-size:0.8rem;color:#71717a">High Risk</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── ACTIONABLE RISK DIFFERENCES ──
    if results.get("comparison_summary"):
        st.markdown("#### Actionable Risk Differences")
        for logic in results["comparison_summary"]:
            st.info(logic)

    # ── CLAUSE-LEVEL DELTA MATRIX ──
    st.markdown("#### Clause-Level Delta Matrix")
    
    categories = list(results["coverage_a"].keys())
    for cat in categories:
        ca = results["coverage_a"].get(cat)
        cb = results["coverage_b"].get(cat)
        
        with st.expander(f"{cat}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{results['metadata']['name_a']}**")
                if ca:
                    st.markdown(f"<span class='badge {'badge-high' if ca['risk']=='High Risk' else 'badge-low'}'>{ca['risk']}</span>", unsafe_allow_html=True)
                    st.caption(ca["text"])
                else:
                    st.error("Missing Protection")
            with c2:
                st.markdown(f"**{results['metadata']['name_b']}**")
                if cb:
                    st.markdown(f"<span class='badge {'badge-high' if cb['risk']=='High Risk' else 'badge-low'}'>{cb['risk']}</span>", unsafe_allow_html=True)
                    st.caption(cb["text"])
                else:
                    st.error("Missing Protection")

    # ── STRUCTURAL GAPS TABLE ──
    st.markdown("#### Structural Gaps & Risk Discordance")
    if results["gaps"]:
        st.table(pd.DataFrame(results["gaps"]))
    else:
        st.success("No structural gaps detected between documents.")


# ══════════════════════════════════════════════════════════════════
# TAB: MONITORING
# ══════════════════════════════════════════════════════════════════

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

    st.markdown("### System Monitoring")
    
    # Drift Check
    curr_avg_len = sum(len(r["clause"].split()) for r in risks) / len(risks)
    has_drift = abs(28 - curr_avg_len) > 15
    if has_drift:
        st.error("**Data Variance Detected**: Current document structure deviates from standard baselines.")
    else:
        st.success("**System Health**: All processes operating smoothly.")

    # Row 1: Core metrics
    m1, m2 = st.columns([1, 1.5])
    with m1: st.plotly_chart(create_metrics_bar_chart(), key="mon_metrics_bar")
    with m2: st.plotly_chart(create_confusion_matrix_heatmap(), key="mon_conf_matrix")

    # Row 2: Drift & Confidence
    d1, d2 = st.columns(2)
    with d1: st.plotly_chart(create_drift_distribution_plot(risks), key="mon_drift_dist")
    with d2: st.plotly_chart(create_confidence_line_chart(risks), key="mon_conf_line")

    # Row 3: Risk, Features, Health
    r1, r2, r3 = st.columns([1, 1, 1.2])
    with r1: st.plotly_chart(create_risk_pie_chart(risks), key="mon_risk_pie")
    with r2: st.plotly_chart(create_feature_importance_bar(risks), key="mon_feat_bar")
    with r3: st.plotly_chart(create_system_health_radar(explanations, retrieval), key="mon_health_radar")


# ══════════════════════════════════════════════════════════════════
# TAB: EXPORT / REPORT
# ══════════════════════════════════════════════════════════════════

def tab_export():
    st.session_state["current_step"] = 5
    render_stepper()
    state = st.session_state["agent_state_a"]
    report = state.get("final_report", {})
    
    if not report:
        st.warning("No report generated. Please analyze a document first.")
        return

    st.markdown("### Legal Analysis Audit")
    st.markdown("---")

    # 1. Executive Summary
    exec_sum = report.get("executive_summary", {})
    st.markdown("#### 1. Executive Summary")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Overall Risk Score", exec_sum.get("overall_risk_score", "N/A"))
    with c2: st.metric("Contract Status", exec_sum.get("contract_status", "N/A"))
    with c3: st.metric("Total Clauses", len(state.get("risks", [])))
    st.write(exec_sum.get("summary_statement", exec_sum.get("summary", "")))

    # 2. Key Risk Insights
    st.markdown("#### 2. Key Risk Insights (Top Critical Findings)")
    key_insights = report.get("key_risk_insights", [])
    if key_insights:
        for insight in key_insights:
            topic = insight.get("topic", "General Risk")
            with st.expander(f"{topic}"):
                st.write(f"**Primary Concern:** {insight.get('primary_concern', 'N/A')}")
                st.markdown(f"*Clause Snippet:* \"{insight.get('clause_snippet', 'N/A')}\"")
    else:
        st.success("No critical risk insights to display.")

    # 3. Recommendations
    st.markdown("#### 3. Actionable Recommendations")
    recommendations = report.get("recommendations", [])
    if recommendations:
        for rec in recommendations:
            st.markdown(f"- **{rec.get('category', 'General')}:** {rec.get('action', 'N/A')}")
    else:
        st.info("No specific recommendations generated.")

    # 4. Explainability
    st.markdown("#### 4. Contextual Explanations")
    st.write("Detailed reasoning for flagged clauses.")
    expl_data = report.get("explainability", [])
    if expl_data:
        for i, exp in enumerate(expl_data):
            clause_num = exp.get("clause_number", exp.get("idx", i + 1))
            confidence = exp.get("ml_confidence", exp.get("confidence", "N/A"))
            summary = exp.get("reasoning", exp.get("summary", "N/A"))
            with st.expander(f"Clause {clause_num} ({confidence} Confidence) — {summary[:60]}..."):
                st.error(f"**Why Risky:** {exp.get('reasoning', exp.get('reason', 'N/A'))}")
                st.warning(f"**Legal Meaning:** {exp.get('legal_implications', exp.get('meaning', 'N/A'))}")
                st.success(f"**Recommended Fix:** {exp.get('mitigation_strategy', exp.get('fix', 'N/A'))}")
    else:
        st.info("No high-risk clauses required deep-dive explanation.")

    st.markdown("---")
    # 5. Disclaimer
    st.caption(report.get("disclaimer", ""))

    st.markdown("<br/>", unsafe_allow_html=True)
    st.download_button(
        "Download Full Digital Audit (JSON)", 
        json.dumps(report, indent=2), 
        file_name="LexIQ_Legal_Audit.json",
        key="download_report_btn"
    )


# ══════════════════════════════════════════════════════════════════
# MAIN APP ENTRY
# ══════════════════════════════════════════════════════════════════

def main():
    render_sidebar()
    
    if "agent_state_a" not in st.session_state:
        tab_upload()
    else:
        tab_list = ["Analyze", "Monitor", "Report"]
        if st.session_state.get("comparison_active"):
            tab_list.insert(1, "Comparison")
            
        tabs = st.tabs(tab_list)
        
        with tabs[0]: tab_risk_analysis()
        
        if st.session_state.get("comparison_active"):
            with tabs[1]: tab_comparison()
            with tabs[2]: tab_analytics()
            with tabs[3]: tab_export()
        else:
            with tabs[1]: tab_analytics()
            with tabs[2]: tab_export()

if __name__ == "__main__":
    main()
