import streamlit as st
import pandas as pd
import pdfplumber
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Structural Path Alignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.clause_segmenter import segment_clauses
from models.inference import risk_engine
from config.settings import RISK_COLORS

# --- DESIGN TOKENS (ULTRA ENTERPRISE V7) ---
COLOR_BG = "#F8FAFC"
COLOR_SURFACE = "#FFFFFF"
COLOR_PRIMARY = "#2563EB"
COLOR_CRITICAL = "#B91C1C"
COLOR_MODERATE = "#D97706"
COLOR_LOW = "#15803D"
COLOR_TEXT = "#0F172A"
COLOR_BORDER = "#E2E8F0"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Legal Intelligence Command Center",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL STYLES ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
    }}
    
    /* Executive Card Styling */
    .kpi-card {{
        background-color: {COLOR_SURFACE};
        padding: 24px;
        border-radius: 4px;
        border: 1px solid {COLOR_BORDER};
        display: flex;
        flex-direction: column;
    }}
    
    .kpi-label {{
        color: #64748B;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 8px;
    }}
    
    .kpi-value {{
        color: {COLOR_TEXT};
        font-size: 1.5rem;
        font-weight: 700;
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 32px;
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 48px;
        background: none;
        border: none;
        color: #64748B;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {COLOR_PRIMARY} !important;
        border-bottom: 2px solid {COLOR_PRIMARY} !important;
    }}
    
    /* Expander Styling */
    div[data-testid="stExpander"] {{
        background-color: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        margin-bottom: 12px;
    }}
    
    /* Code block styling */
    code {{
        background-color: #F1F5F9 !important;
        color: #334155 !important;
        padding: 8px !important;
        border-radius: 4px;
    }}
</style>
""", unsafe_allow_html=True)

# --- MODULAR RENDERING FUNCTIONS ---

def render_control_panel():
    """Refined sidebar control panel."""
    with st.sidebar:
        st.markdown(f"<h3 style='font-weight: 700; margin-bottom: 4px;'>Platform Console</h3>", unsafe_allow_html=True)
        st.caption("Self-Healing Autonomous Legal AI")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.write("**System Status**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Risk Engine")
            st.markdown("<p style='color: #15803D; font-size: 0.75rem; font-weight: 600;'>● OPERATIONAL</p>", unsafe_allow_html=True)
        with col2:
            st.caption("Agent Node")
            st.markdown("<p style='color: #64748B; font-size: 0.75rem; font-weight: 600;'>○ STANDBY</p>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.write("**Contract Intake**")
        uploaded_file = st.file_uploader("Upload legal document (PDF, TXT)", type=["pdf", "txt"], label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.write("**Model Intelligence**")
        st.caption("Balanced Logistic Regression V1")
        st.caption("TF-IDF Clause Vectorization")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.write("**Operational Controls**")
        st.checkbox("Autonomous Mitigation (M2)", disabled=True)
        st.checkbox("Policy Enforcement (M2)", disabled=True)
        
        return uploaded_file

def render_header_banner(file_present):
    """Quiet, authoritative header banner."""
    st.markdown(f"<h1 style='font-weight: 700; font-size: 1.875rem; margin-bottom: 8px;'>Legal Intelligence Command Center</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size: 1rem;'>AI-assisted contract risk analysis for internal review workflows.</p>", unsafe_allow_html=True)
    
    st.caption("Governance Notice: This system provides analytical assistance and does not replace professional legal judgment.")
    st.markdown("---")

def render_executive_overview(df):
    """Professional KPI overview for executive review."""
    total = len(df)
    high_risk = len(df[df['Risk Level'] == "High Risk"])
    exposure = (high_risk / total * 100) if total > 0 else 0
    confidence = df['Confidence'].mean() * 100 if total > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    
    metrics = [
        ("Total Clauses Reviewed", f"{total}"),
        ("High-Risk Flags", f"{high_risk}"),
        ("Risk Exposure Index", f"{exposure:.1f}%"),
        ("Model Confidence", f"{confidence:.1f}%")
    ]
    
    for i, (label, value) in enumerate(metrics):
        with [c1, c2, c3, c4][i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

def render_risk_radar_panel(df):
    """Strategic radar with minimalist chart styling."""
    st.markdown("<h3 style='font-weight: 600; font-size: 1.25rem;'>Strategic Posture Analysis</h3>", unsafe_allow_html=True)
    
    if not df.empty:
        counts = df['Risk Level'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 2), facecolor=COLOR_BG)
        
        # Professional color palette
        colors = [COLOR_CRITICAL if x == "High Risk" else COLOR_LOW for x in counts.index]
        
        sns.barplot(x=counts.values, y=counts.index, palette=colors, ax=ax, orient='h')
        
        ax.set_facecolor(COLOR_BG)
        ax.tick_params(colors="#64748B", labelsize=8)
        sns.despine(left=True, bottom=True)
        ax.grid(axis='x', color=COLOR_BORDER, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)
    else:
        st.info("Awaiting strategic data ingestion.")

def render_clause_workspace(df):
    """Human-readable review workspace with semantic hierarchy."""
    st.markdown("<h3 style='font-weight: 600; font-size: 1.25rem; margin-top: 24px;'>Operational Clause Audit</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size: 0.875rem; margin-bottom: 16px;'>Identified provisions requiring manual oversight and mitigating action.</p>", unsafe_allow_html=True)
    
    for i, row in df.iterrows():
        label = row['Risk Level']
        is_high = label == "High Risk"
        
        title = f"{'Critical' if is_high else 'Standard'} | Clause {i+1} | Integrity: {row['Confidence']:.1%}"
        
        with st.expander(title, expanded=is_high):
            st.markdown("**Instrument Segment**")
            st.markdown(f"```text\n{row['Clause']}\n```")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Semantic feedback container
            bg_color = "#FEF2F2" if is_high else "#F0FDF4"
            border_color = COLOR_CRITICAL if is_high else COLOR_LOW
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; border-left: 4px solid {border_color}; padding: 16px; border-radius: 4px;">
                <p style="font-weight: 600; margin-bottom: 4px; color: {COLOR_TEXT};">AI Analysis Summary</p>
                <p style="font-size: 0.875rem; color: #475569;">
                    { 'Significant legal exposure detected. Triggers suggest potential liability or compliance conflict.' if is_high else 'Operational language verified as low-risk standard provision.' }
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if row['Keywords']:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Detection Vectors**")
                cols = st.columns(len(row['Keywords']) if len(row['Keywords']) > 0 else 1)
                for idx, word in enumerate(row['Keywords']):
                    with cols[min(idx, len(cols)-1)]:
                        st.markdown(f"<span style='background-color: #E2E8F0; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; color: #475569;'>{word}</span>", unsafe_allow_html=True)

def render_intelligence_analytics(df):
    """Detailed research hub for deep-dive analysis."""
    st.markdown("<h3 style='font-weight: 600; font-size: 1.25rem; margin-top: 32px;'>Intelligence Research Hub</h3>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Posture", "Stability Trend", "Keyword Matrix", "Agent Intelligence"])
    
    with tab1:
        if not df.empty:
            st.bar_chart(df['Risk Level'].value_counts())
        else:
            st.caption("Landscape data unavailable.")
            
    with tab2:
        if not df.empty:
            st.line_chart(df['Confidence'])
            st.caption("Distribution of model integrity across all analyzed segments.")
            
    with tab3:
        if not df.empty:
            st.caption("High-frequency trigger signals identified across document corpus.")
            # Simple keyword list
            all_kw = []
            for kws in df['Keywords']: all_kw.extend(kws)
            if all_kw:
                st.write(", ".join(list(set(all_kw))[:20]))
                
    with tab4:
        st.info("Conceptual Foundation: Agentic reasoning pipelines are scheduled for Milestone-2 deployment. This module will eventually host RAG-based legal guidance.")

def render_governance_section():
    """Model governance and compliance integrity."""
    st.markdown("---")
    st.markdown("<h3 style='font-weight: 600; font-size: 1.25rem;'>Model Governance</h3>", unsafe_allow_html=True)
    
    g1, g2 = st.columns([1, 1.5])
    
    with g1:
        st.markdown("**Foundational Architecture**")
        st.caption("- Pipeline: spaCy NLP L3")
        st.caption("- Classifier: Balanced Logistic Regression")
        st.caption("- Optimization: TF-IDF Weighted Vectors")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Recall vs. Precision**")
        st.p(
            "In legal risk detection, we prioritize **Recall**. Our goal is to ensure that no critical provision is "
            "missed, even if it leads to occasional conservative flags for manual review."
        )
        
    with g2:
        st.markdown("**Performance Integrity (Confusion Matrix)**")
        try:
            st.image("artifacts/logistic_regression_matrix.png", use_container_width=True)
        except:
            st.caption("Evaluation artifacts unavailable in current environment.")

def render_onboarding_state():
    """Calm onboarding state for new user sessions."""
    st.markdown(f"""
    <div style="background-color: #FFFFFF; border: 1px solid {COLOR_BORDER}; padding: 40px; border-radius: 4px; text-align: center;">
        <h2 style="font-weight: 700; color: {COLOR_TEXT};">Initialize Platform Analysis</h2>
        <p style="color: #64748B; margin-bottom: 24px;">Please provide a legal instrument in the control panel to begin the risk decomposition process.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Capabilities Overview**")
    
    c1, c2, c3 = st.columns(3)
    features = [
        ("Automated Parsing", "Segment long documents into discrete, reviewable clauses."),
        ("Risk Classification", "Identify high-exposure legal language with statistical confidence."),
        ("Keyword Vectorization", "Extract semantic triggers driving the risk assessment.")
    ]
    
    for i, (title, desc) in enumerate(features):
        with [c1, c2, c3][i]:
            st.markdown(f"**{title}**")
            st.caption(desc)

# --- UTILITY: FILE PARSING ---
def get_text_from_file(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        return file.getvalue().decode("utf-8")

# --- MAIN OPERATIONAL FLOW ---

def main():
    # 1. Control Panel
    uploaded_file = render_control_panel()
    
    # 2. Header
    render_header_banner(bool(uploaded_file))
    
    if uploaded_file:
        # 3. Processing (Self-Healing Overlay)
        with st.spinner("Analyzing document structure and risk vectors..."):
            try:
                raw_text = get_text_from_file(uploaded_file)
                clauses = segment_clauses(raw_text)
                
                results = []
                for idx, c in enumerate(clauses):
                    label, conf, reasons = risk_engine.analyze_clause(c)
                    results.append({
                        "id": idx,
                        "Clause": c,
                        "Risk Level": label,
                        "Confidence": conf,
                        "Keywords": reasons
                    })
                df = pd.DataFrame(results)
                
                # Success state
                st.toast("Analysis and vectorization complete.")
                
                # 4. Rendering Modules
                render_executive_overview(df)
                
                col_radar, col_spacer = st.columns([2, 1])
                with col_radar:
                    render_risk_radar_panel(df)
                
                render_clause_workspace(df)
                render_intelligence_analytics(df)
                render_governance_section()
                
            except Exception as e:
                st.error("Platform Error: An inconsistency occurred during document parsing. Accessing recovery protocols.")
                st.caption(f"Technical detail: {str(e)}")
    else:
        render_onboarding_state()

if __name__ == "__main__":
    main()
