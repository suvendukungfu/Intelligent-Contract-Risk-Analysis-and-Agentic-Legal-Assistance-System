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

# --- BRANDING & DESIGN TOKENS ---
COLOR_BG = "#F1F5F9"
COLOR_SURFACE = "#FFFFFF"
COLOR_BLUE = "#1D4ED8" # Command Blue
COLOR_PURPLE = "#7C3AED" # Autonomous Accent
COLOR_RED = "#B91C1C" # Critical Risk
COLOR_GREEN = "#15803D" # Secure Green
COLOR_BORDER = "#CBD5E1"
COLOR_TEXT = "#0F172A"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Autonomous Legal Command Bridge",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL STYLES ---
st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    .stMetric {{ background-color: {COLOR_SURFACE}; padding: 15px; border-radius: 8px; border: 1px solid {COLOR_BORDER}; }}
    div[data-testid="stExpander"] {{ background-color: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; white-space: pre-wrap; }}
    
    /* Executive Badge Styles */
    .pill {{
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }}
    .pill-blue {{ background-color: #DBEAFE; color: {COLOR_BLUE}; }}
    .pill-purple {{ background-color: #F3E8FF; color: {COLOR_PURPLE}; }}
    .pill-green {{ background-color: #DCFCE7; color: {COLOR_GREEN}; }}
</style>
""", unsafe_allow_html=True)

# --- COMPOSABLE UI LAYERS ---

def render_command_dock_sidebar():
    """Redesign sidebar into layered operational zones."""
    with st.sidebar:
        st.markdown(f"<h2 style='color: {COLOR_BLUE};'>⚖️ Legal Guard AI</h2>", unsafe_allow_html=True)
        st.markdown("<span class='pill pill-blue'>Autonomous Command Bridge Mode</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📂 Contract Intake")
        uploaded_file = st.file_uploader("Drop legal instrument here", type=["pdf", "txt"], label_visibility="collapsed")
        
        if uploaded_file:
            st.success("Instrument Locked 🔒")
        else:
            st.info("Awaiting Input Data...")

        st.markdown("---")
        st.subheader("🧠 AI Engine Status")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.caption("Classifier")
            st.markdown("<span style='color: green;'>● ACTIVE</span>", unsafe_allow_html=True)
        with col_s2:
            st.caption("Agent Engine")
            st.markdown("<span style='color: orange;'>○ STANDBY</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔄 Workflow Controls")
        st.checkbox("Enable Agent Reasoning (Milestone-2)", disabled=True, help="Agentic workflows are currently in roadmap development.")
        
        return uploaded_file

def render_bridge_header(file_present):
    """Full-width strategic command banner."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Autonomous Legal Command Bridge")
        st.markdown("AI-Augmented Contract Intelligence Platform & Risk Monitoring Console")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if file_present:
            st.markdown("<span class='pill pill-green'>Status: Analysis Ready</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='pill pill-blue'>Status: System Ready</span>", unsafe_allow_html=True)
        st.markdown("🔍 **Persona:** Paralegal Assistant")

    st.caption("⚠️ Governance Protocol: AI assists legal professionals — human review required for all risk finalizations.")
    st.markdown("---")

def render_command_strip(df):
    """Upgrade KPI ribbon into operational command indicators."""
    st.subheader("Executive Command Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    total = len(df)
    critical = len(df[df['Risk Level'] == "High Risk"])
    exposure = (critical / total) * 100 if total > 0 else 0
    trust = df['Confidence'].mean() * 100 if total > 0 else 0
    
    c1.metric("Total Clauses", total)
    c2.metric("Critical Alerts", critical, delta=f"{exposure:.1f}% Exposure", delta_color="inverse" if critical > 0 else "off")
    c3.metric("Exposure Index", f"{exposure:.1f}%")
    c4.metric("Trust Stability", f"{trust:.1f}%")
    c5.metric("Agent Readiness", "Milestone 1", delta="M2 Pending")

def render_risk_radar(df):
    """Modular radar equivalent visualization for future agent update."""
    st.subheader("Autonomous Risk Radar")
    if not df.empty:
        counts = df['Risk Level'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 3), facecolor=COLOR_BG)
        sns.barplot(x=counts.values, y=counts.index, palette="magma", ax=ax, orient='h')
        ax.set_facecolor(COLOR_BG)
        ax.set_title("Clause Distribution by Strategic Risk Weight", color=COLOR_TEXT)
        st.pyplot(fig)
    else:
        st.info("Radar initialized. Awaiting data parsing...")

def render_agent_workflow_preview():
    """Future-ready visual scaffolding for Milestone-2."""
    st.markdown(f"<h3 style='color: {COLOR_PURPLE};'>🧠 Autonomous Reasoning Pipeline (Preview)</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown("#### Step 1\n**Risk Detection**")
        st.success("● ACTIVE")
    with c2:
        st.markdown("#### Step 2\n**Legal Retrieval**")
        st.warning("○ Coming Soon")
    with c3:
        st.markdown("#### Step 3\n**Reasoning Agent**")
        st.info("⌛ Milestone-2")
    with c4:
        st.markdown("#### Step 4\n**Report Gen**")
        st.info("⌛ Planned")
    st.markdown("<br>", unsafe_allow_html=True)

def render_intelligence_hub(df):
    """Enhanced analytics hub prepared for agent logic expansion."""
    st.subheader("Intelligence Analytics Hub")
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Landscape", "Confidence Stability", "Keyword Intelligence", "Agent Insights (M2)"])
    
    with tab1:
        if not df.empty:
            st.bar_chart(df['Risk Level'].value_counts())
        else:
            st.write("Awaiting document parsing for landscape visualization.")
            
    with tab2:
        if not df.empty:
            st.line_chart(df['Confidence'])
            st.caption("Temporal confidence stability across clause segments.")
            
    with tab3:
        if not df.empty:
            all_keywords = []
            for kw in df['Keywords']:
                all_keywords.extend(kw)
            if all_keywords:
                kw_counts = pd.Series(all_keywords).value_counts().head(10)
                st.table(pd.DataFrame({"Trigger Word": kw_counts.index, "Frequency": kw_counts.values}))
            else:
                st.write("No trigger keywords detected.")
                
    with tab4:
        st.info("Agentic insights will be available once the Reasoning Engine (Milestone-2) is activated.")

def render_operational_review_panels(df):
    """Renamed clause audit sections with enhanced UI elements."""
    st.subheader("Operational Legal Review Panels")
    st.caption("AI-identified triggers requiring tactical human oversight.")
    
    for i, row in df.iterrows():
        label = row['Risk Level']
        color = COLOR_RED if label == "High Risk" else COLOR_GREEN
        icon = "🚨" if label == "High Risk" else "🛡️"
        
        with st.expander(f"{icon} {label.upper()} | CID-{1000+i} | Trust: {row['Confidence']:.1%}", expanded=(label == "High Risk")):
            st.markdown(f"**Clause Content:**")
            st.code(row['Clause'], language="text")
            
            # AI Tactical Insight
            st.markdown(f"<div style='border-left: 4px solid {color}; padding-left: 15px; margin-top: 10px;'>", unsafe_allow_html=True)
            if label == "High Risk":
                st.markdown(f"**Tactical Insight:** High-priority exposure detected. Triggers suggest significant liability or restrictive governance.")
                if row['Keywords']:
                    st.write("**Specific Trigger Tags:**")
                    st.write(" ".join([f"`{w}`" for w in row['Keywords']]))
            else:
                st.markdown("**Tactical Insight:** Standard operational language. Low immediate risk signature.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Confidence Meter
            st.progress(float(row['Confidence']))

def render_governance_console():
    """Footer console for platform integrity and future governance."""
    st.markdown("---")
    st.subheader("Autonomous Governance Center")
    t1, t2, t3 = st.tabs(["Model Foundations", "Performance Integrity", "Future Agent Governance"])
    
    with t1:
        st.write("**Architecture:** Supervised Learning via Balanced Logistic Regression")
        st.write("**NLP Core:** spaCy v3.x Pipeline")
        
    with t2:
        st.write("**Model Accuracy Matrix**")
        try:
            st.image("artifacts/logistic_regression_matrix.png", width=600)
            st.caption("Metrics verify model's Recall stability—essential for ensuring no critical risk is omitted.")
        except:
            st.error("Platform Health Check: Evaluation artifacts not found. Please sync local brain.")
            
    with t3:
        st.info("Governance frameworks for Milestone-2 Agents (Human-in-the-loop, Auditability) are currently being drafted.")

def render_bridge_empty_state():
    """Onboarding interface for the command bridge."""
    st.markdown("### Initialize Autonomous Legal Command Bridge")
    st.write("Upload a contract in the Command Dock to initialize analysis and agentic mapping.")
    
    # Mock visual preview
    col1, col2 = st.columns(2)
    with col1:
        st.info("Step 1: Parse Document")
    with col2:
        st.info("Step 2: Map Autonomous Vectors")
    
    example_data = {
        "Strategic Category": ["Liabilities", "Governance", "Operational"],
        "Risk Signature": ["High", "Medium", "Neutral"],
        "Agent Readiness": ["Detected", "Coming Soon", "Planned"]
    }
    st.table(example_data)

# --- UTILITY: FILE PARSER ---
def get_text_from_file(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        return file.getvalue().decode("utf-8")

# --- MAIN EXECUTION FLOW ---

def main():
    # 1. Sidebar / Command Dock
    uploaded_file = render_command_dock_sidebar()
    
    # 2. Header
    render_bridge_header(bool(uploaded_file))
    
    if uploaded_file:
        # 3. Core Processing (Milestone-1 Engine)
        with st.status("AI Command Bridge Processing...", expanded=True) as status:
            raw_text = get_text_from_file(uploaded_file)
            st.write("Segmenting Clauses...")
            clauses = segment_clauses(raw_text)
            
            st.write("Running Neural Analytics...")
            results = []
            for c in clauses:
                label, conf, reasons = risk_engine.analyze_clause(c)
                results.append({
                    "Clause": c, 
                    "Risk Level": label, 
                    "Confidence": conf,
                    "Keywords": reasons
                })
            df = pd.DataFrame(results)
            status.update(label="Command Bridge Updated!", state="complete", expanded=False)
        
        st.toast("Command Bridge Updated")
        
        # 4. Content Rendering
        render_command_strip(df)
        
        col_main, col_viz = st.columns([1.5, 1])
        with col_main:
            render_agent_workflow_preview()
        with col_viz:
            render_risk_radar(df)
            
        render_operational_review_panels(df)
        render_intelligence_hub(df)
        render_governance_console()
        
    else:
        # 5. Empty State
        render_bridge_empty_state()

if __name__ == "__main__":
    main()
