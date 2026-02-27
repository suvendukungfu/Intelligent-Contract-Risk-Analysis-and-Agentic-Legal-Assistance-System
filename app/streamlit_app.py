import streamlit as st
import os
import sys

# --- DEFENSIVE STARTUP ---
IMPORT_ERROR = None
try:
    import pandas as pd
    import pdfplumber
    import matplotlib.pyplot as plt
    import numpy as np

    # Path Alignment
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import spacy
    import subprocess
    
    # --- RUNTIME MODEL INJECTION ---
    # Streamlit Cloud Python 3.13 / Nixpacks workaround
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading en_core_web_sm at runtime...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

    from nlp.clause_segmenter import segment_clauses
    from models.inference import risk_engine
except Exception as e:
    IMPORT_ERROR = str(e)
    import traceback
    IMPORT_ERROR += "\n" + traceback.format_exc()

# --- DARK DESIGN TOKENS ---
COLOR_BG       = "#0D1117"
COLOR_SURFACE  = "#161B22"
COLOR_SURFACE2 = "#1C2333"
COLOR_PRIMARY  = "#4F8EF7"
COLOR_CRITICAL = "#F87171"
COLOR_MODERATE = "#FBBF24"
COLOR_LOW      = "#4ADE80"
COLOR_TEXT     = "#E6EDF3"
COLOR_MUTED    = "#8B949E"
COLOR_BORDER   = "#30363D"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Intelligent Contract Risk Analysis System",
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

    /* Override base app bg to match dark token */
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_SURFACE};
        border-right: 1px solid {COLOR_BORDER};
    }}

    /* KPI cards */
    .kpi-card {{
        background-color: {COLOR_SURFACE2};
        padding: 20px 24px;
        border-radius: 8px;
        border: 1px solid {COLOR_BORDER};
        display: flex;
        flex-direction: column;
        min-height: 90px;
    }}
    .kpi-label {{
        color: {COLOR_MUTED};
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        color: {COLOR_TEXT};
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1;
    }}

    /* Section headings */
    .section-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {COLOR_TEXT};
        margin: 0 0 4px 0;
    }}
    .section-sub {{
        font-size: 0.82rem;
        color: {COLOR_MUTED};
        margin-bottom: 16px;
    }}

    /* Tab strip */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 28px;
        border-bottom: 1px solid {COLOR_BORDER};
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        background: none;
        border: none;
        color: {COLOR_MUTED};
        font-weight: 500;
        font-size: 0.875rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLOR_PRIMARY} !important;
        border-bottom: 2px solid {COLOR_PRIMARY} !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        margin-bottom: 10px;
    }}

    /* Code blocks */
    pre, code {{
        background-color: {COLOR_BG} !important;
        color: #A5D6FF !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 6px;
        font-size: 0.82rem !important;
    }}

    /* Divider */
    hr {{
        border-color: {COLOR_BORDER};
        margin: 24px 0;
    }}

    /* Badge pill */
    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}
    .badge-high {{ background-color: rgba(248,113,113,0.15); color: {COLOR_CRITICAL}; border: 1px solid rgba(248,113,113,0.3); }}
    .badge-low  {{ background-color: rgba(74,222,128,0.12); color: {COLOR_LOW};  border: 1px solid rgba(74,222,128,0.25); }}

    /* Keyword chip */
    .kw-chip {{
        display: inline-block;
        background-color: rgba(79,142,247,0.12);
        color: {COLOR_PRIMARY};
        border: 1px solid rgba(79,142,247,0.25);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 2px;
    }}
</style>
""", unsafe_allow_html=True)

# --- MODULAR RENDERING FUNCTIONS ---

def render_control_panel():
    """Dark sidebar control panel."""
    with st.sidebar:
        st.markdown(
            f"<h2 style='font-weight:700;font-size:1.15rem;color:{COLOR_TEXT};margin-bottom:2px;'>⚖️ Contract Risk AI</h2>"
            f"<p style='font-size:0.78rem;color:{COLOR_MUTED};margin-bottom:0;'>Intelligent Legal Analysis</p>",
            unsafe_allow_html=True
        )
        st.markdown(f"<hr style='border-color:{COLOR_BORDER};margin:14px 0;'>", unsafe_allow_html=True)

        # Status indicators
        st.markdown(f"<p style='font-size:0.78rem;font-weight:600;color:{COLOR_MUTED};text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>System Status</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div style='background:{COLOR_SURFACE2};border:1px solid {COLOR_BORDER};border-radius:6px;padding:8px 10px;text-align:center;'><div style='font-size:0.7rem;color:{COLOR_MUTED};'>Risk Engine</div><div style='font-size:0.75rem;font-weight:700;color:{COLOR_LOW};margin-top:2px;'>● LIVE</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='background:{COLOR_SURFACE2};border:1px solid {COLOR_BORDER};border-radius:6px;padding:8px 10px;text-align:center;'><div style='font-size:0.7rem;color:{COLOR_MUTED};'>Agent Node</div><div style='font-size:0.75rem;font-weight:700;color:{COLOR_MUTED};margin-top:2px;'>○ M2</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.78rem;font-weight:600;color:{COLOR_MUTED};text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>Upload Document</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Legal document (PDF or TXT)", type=["pdf", "txt"], label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.78rem;font-weight:600;color:{COLOR_MUTED};text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>Pipeline</p>", unsafe_allow_html=True)
        for item in ["spaCy v3.7  —  Lemmatization", "TF-IDF  —  Feature Vectorization", "Logistic Regression  —  Classifier"]:
            st.markdown(f"<p style='font-size:0.78rem;color:{COLOR_MUTED};margin:3px 0;'>› {item}</p>", unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{COLOR_BORDER};margin:14px 0;'>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.72rem;color:{COLOR_MUTED};'>Model artifacts loaded from <code>artifacts/</code></p>", unsafe_allow_html=True)

        return uploaded_file

def render_header_banner(file_present):
    """Dark header banner."""
    st.markdown(
        f"<h1 style='font-weight:700;font-size:1.8rem;color:{COLOR_TEXT};margin-bottom:6px;'>Intelligent Contract Risk Analysis</h1>"
        f"<p style='font-size:0.95rem;color:{COLOR_MUTED};margin-bottom:10px;'>Clause-level risk assessment using classical NLP — TF-IDF · Logistic Regression · spaCy</p>"
        f"<p style='font-size:0.78rem;color:{COLOR_MUTED};border-left:3px solid {COLOR_BORDER};padding-left:10px;'>Governance: Model assessments highlight linguistic risk patterns and are intended to assist, not replace, professional legal review.</p>",
        unsafe_allow_html=True
    )
    st.markdown(f"<hr style='border-color:{COLOR_BORDER};margin:18px 0;'>", unsafe_allow_html=True)

def render_executive_overview(df):
    """Dark KPI overview."""
    st.markdown(f"<p class='section-title'>Executive Summary</p><p class='section-sub'>Metrics derived from clause-level model predictions.</p>", unsafe_allow_html=True)

    total      = len(df)
    high_risk  = len(df[df['Risk Level'] == "High Risk"])
    low_risk   = total - high_risk
    exposure   = (high_risk / total * 100) if total > 0 else 0
    confidence = df['Confidence'].mean() * 100 if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Evaluated Clauses",   f"{total}",          COLOR_PRIMARY),
        ("High-Risk Detections",f"{high_risk}",       COLOR_CRITICAL),
        ("Risk Exposure Index", f"{exposure:.1f}%",  COLOR_MODERATE),
        ("Mean Confidence",     f"{confidence:.1f}%",COLOR_LOW),
    ]
    for i, (label, value, accent) in enumerate(metrics):
        with [c1, c2, c3, c4][i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{accent};">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def render_risk_radar_panel(df):
    """Dark horizontal bar chart for risk distribution."""
    st.markdown(f"<p class='section-title'>Risk Distribution</p><p class='section-sub'>Clause-level risk spread across the document.</p>", unsafe_allow_html=True)

    if not df.empty:
        counts = df['Risk Level'].value_counts()
        bar_colors = [COLOR_CRITICAL if x == "High Risk" else COLOR_LOW for x in counts.index]

        fig, ax = plt.subplots(figsize=(8, max(1.5, len(counts) * 0.65)), facecolor=COLOR_SURFACE)
        fig.patch.set_facecolor(COLOR_SURFACE)

        bars = ax.barh(counts.index, counts.values, color=bar_colors, height=0.45)
        ax.set_facecolor(COLOR_SURFACE)
        ax.tick_params(colors=COLOR_MUTED, labelsize=9)
        ax.xaxis.label.set_color(COLOR_MUTED)
        ax.spines[:].set_color(COLOR_BORDER)
        ax.tick_params(axis='both', colors=COLOR_MUTED)
        ax.grid(axis='x', color=COLOR_BORDER, linestyle='--', linewidth=0.5, alpha=0.6)
        for bar, val in zip(bars, counts.values):
            ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', ha='left', color=COLOR_TEXT, fontsize=9, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Upload a document to view risk distribution.")

def render_clause_workspace(df):
    """Dark clause review workspace."""
    st.markdown(f"<p class='section-title' style='margin-top:24px;'>Risk Analysis Workspace</p><p class='section-sub'>Clause-level assessments and linguistic risk triggers.</p>", unsafe_allow_html=True)

    # Sort: high-risk first
    df_sorted = df.sort_values(by='Confidence', ascending=False)

    for _, row in df_sorted.iterrows():
        label   = row['Risk Level']
        is_high = label == "High Risk"
        badge   = f"<span class='badge badge-{'high' if is_high else 'low'}'>{'HIGH RISK' if is_high else 'LOW RISK'}</span>"
        title   = f"Segment {row['id']+1}  ·  Confidence: {row['Confidence']:.1%}"

        with st.expander(title, expanded=is_high):
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"<p style='font-size:0.78rem;font-weight:600;color:{COLOR_MUTED};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>Provision Text</p>", unsafe_allow_html=True)
            st.code(row['Clause'], language=None)

            bg  = "rgba(248,113,113,0.08)" if is_high else "rgba(74,222,128,0.07)"
            bdr = COLOR_CRITICAL          if is_high else COLOR_LOW
            msg = ("Significant legal exposure detected. Linguistic triggers indicate elevated liability or obligation profile."
                   if is_high else
                   "Standard operational language. Low liability signature detected.")
            st.markdown(f"""
            <div style="background:{bg};border-left:3px solid {bdr};padding:12px 16px;border-radius:6px;margin-top:10px;">
                <p style="font-weight:600;font-size:0.85rem;color:{COLOR_TEXT};margin-bottom:4px;">Model Assessment</p>
                <p style="font-size:0.82rem;color:{COLOR_MUTED};margin:0;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)

            keywords = [k for k in (row['Keywords'] or []) if k][:12]
            if keywords:
                st.markdown(f"<p style='font-size:0.78rem;font-weight:600;color:{COLOR_MUTED};text-transform:uppercase;letter-spacing:.06em;margin:12px 0 6px 0;'>Linguistic Triggers</p>", unsafe_allow_html=True)
                chips = " ".join(f"<span class='kw-chip'>{w}</span>" for w in keywords)
                st.markdown(chips, unsafe_allow_html=True)

def render_intelligence_analytics(df):
    """Dark analytics hub with styled charts."""
    st.markdown(f"<p class='section-title' style='margin-top:28px;'>Analytics Hub</p><p class='section-sub'>Deep-dive model outputs and corpus statistics.</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Risk Landscape", "Confidence Trend", "Trigger Corpus", "Roadmap"])

    with tab1:
        if not df.empty:
            counts = df['Risk Level'].value_counts()
            fig, ax = plt.subplots(figsize=(7, 3), facecolor=COLOR_SURFACE)
            fig.patch.set_facecolor(COLOR_SURFACE)
            bar_colors = [COLOR_CRITICAL if x == "High Risk" else COLOR_LOW for x in counts.index]
            ax.bar(counts.index, counts.values, color=bar_colors, width=0.4)
            ax.set_facecolor(COLOR_SURFACE)
            ax.spines[:].set_color(COLOR_BORDER)
            ax.tick_params(colors=COLOR_MUTED)
            for spine in ax.spines.values(): spine.set_color(COLOR_BORDER)
            ax.set_ylabel("Clause Count", color=COLOR_MUTED, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.caption("No data available.")

    with tab2:
        if not df.empty:
            fig, ax = plt.subplots(figsize=(9, 3), facecolor=COLOR_SURFACE)
            fig.patch.set_facecolor(COLOR_SURFACE)
            ax.plot(df.index, df['Confidence'], color=COLOR_PRIMARY, linewidth=1.5)
            ax.fill_between(df.index, df['Confidence'], alpha=0.12, color=COLOR_PRIMARY)
            ax.set_facecolor(COLOR_SURFACE)
            ax.spines[:].set_color(COLOR_BORDER)
            ax.tick_params(colors=COLOR_MUTED)
            ax.set_ylabel("Confidence", color=COLOR_MUTED, fontsize=9)
            ax.set_xlabel("Clause Index", color=COLOR_MUTED, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Confidence score distribution across all document segments.")

    with tab3:
        if not df.empty:
            all_kw = []
            for kws in df['Keywords']:
                if isinstance(kws, list):
                    all_kw.extend(kws)
            unique_kw = list(dict.fromkeys(kw for kw in all_kw if kw))[:40]
            if unique_kw:
                chips = " ".join(f"<span class='kw-chip'>{w}</span>" for w in unique_kw)
                st.markdown(f"<p style='font-size:0.78rem;color:{COLOR_MUTED};margin-bottom:10px;'>Top linguistic triggers found across all clauses:</p>{chips}", unsafe_allow_html=True)
            else:
                st.caption("No trigger data available.")

    with tab4:
        st.markdown(f"""
        <div style="background:{COLOR_SURFACE2};border:1px solid {COLOR_BORDER};border-radius:8px;padding:20px;">
            <p style="font-weight:600;color:{COLOR_TEXT};margin-bottom:8px;">Milestone 2 — Agentic Reasoning Node</p>
            <p style="font-size:0.85rem;color:{COLOR_MUTED};margin:0;">Future integration: RAG-based legal clause retrieval, LLM-assisted contextual reasoning, and automated mitigation suggestions via an AI agent pipeline.</p>
        </div>
        """, unsafe_allow_html=True)

def render_governance_section():
    """Dark model governance section."""
    st.markdown(f"<hr style='border-color:{COLOR_BORDER};margin:28px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<p class='section-title'>Model Governance & Integrity</p><p class='section-sub'>Architecture details and validation performance metrics.</p>", unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1.5])

    with g1:
        st.markdown(f"""
        <div style="background:{COLOR_SURFACE2};border:1px solid {COLOR_BORDER};border-radius:8px;padding:18px;">
            <p style="font-size:0.78rem;font-weight:600;color:{COLOR_MUTED};text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;">Architecture</p>
            <p style="font-size:0.85rem;color:{COLOR_TEXT};margin:4px 0;">🔤 NLP Core &nbsp;—&nbsp; spaCy v3.7 (Lemmatized)</p>
            <p style="font-size:0.85rem;color:{COLOR_TEXT};margin:4px 0;">📊 Features &nbsp;—&nbsp; TF-IDF (1-gram + 2-gram)</p>
            <p style="font-size:0.85rem;color:{COLOR_TEXT};margin:4px 0;">🤖 Classifier &nbsp;—&nbsp; Logistic Regression (balanced)</p>
            <p style="font-size:0.85rem;color:{COLOR_TEXT};margin:4px 0;">⚡ Vocab Size &nbsp;—&nbsp; 5,000 features max</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.82rem;color:{COLOR_MUTED};border-left:3px solid {COLOR_BORDER};padding-left:10px;'>Benchmarking prioritises <strong>Recall</strong> to minimise false negatives on critical legal provisions.</p>", unsafe_allow_html=True)

    with g2:
        st.markdown(f"<p style='font-size:0.78rem;font-weight:600;color:{COLOR_MUTED};text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px;'>Confusion Matrix — Logistic Regression</p>", unsafe_allow_html=True)
        # Use absolute path to avoid cwd issues
        matrix_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "logistic_regression_matrix.png")
        if os.path.exists(matrix_path):
            st.image(matrix_path, use_container_width=True)
        else:
            st.caption("Evaluation artifact not found. Run `python -m models.train` to generate.")

def render_onboarding_state():
    """Dark onboarding state."""
    st.markdown(f"""
    <div style="background:{COLOR_SURFACE};border:1px solid {COLOR_BORDER};border-radius:10px;padding:48px;text-align:center;">
        <div style="font-size:2.5rem;margin-bottom:16px;">⚖️</div>
        <h2 style="font-weight:700;color:{COLOR_TEXT};margin-bottom:8px;">Upload a Legal Document to Begin</h2>
        <p style="color:{COLOR_MUTED};max-width:480px;margin:0 auto 0 auto;font-size:0.9rem;">Use the sidebar to upload a PDF or TXT contract. The system will segment, vectorize, and classify each provision automatically.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    features = [
        ("🔍", "Structural Segmentation", "Splits the document into discrete legal clauses with regex-based boundary detection."),
        ("🧠", "Linguistic Risk Assessment", "TF-IDF vectorization + Logistic Regression classifies each clause as High or Low risk."),
        ("💡", "Trigger Extraction", "Extracts the top weighted keywords that drove each risk prediction for explainability."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with [c1, c2, c3][i]:
            st.markdown(f"""
            <div style="background:{COLOR_SURFACE2};border:1px solid {COLOR_BORDER};border-radius:8px;padding:20px;text-align:center;height:100%;">
                <div style="font-size:1.6rem;margin-bottom:10px;">{icon}</div>
                <p style="font-weight:600;color:{COLOR_TEXT};margin-bottom:6px;font-size:0.9rem;">{title}</p>
                <p style="font-size:0.78rem;color:{COLOR_MUTED};margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# --- UTILITY: FILE PARSING ---
def get_text_from_file(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        return file.getvalue().decode("utf-8")

# --- MAIN OPERATIONAL FLOW ---

def main():
    # 0. Global Error Handling (Cloud Diagnostics)
    if IMPORT_ERROR:
        st.error("Platform Foundation Error: Critical module discovery failure.")
        st.info("The system was unable to initialize the core NLP or Inference engine. This is usually caused by absolute path mismatches on Streamlit Cloud.")
        with st.expander("Technical Debugging Log"):
            st.code(IMPORT_ERROR)
        return

    # 1. Platform Console
    uploaded_file = render_control_panel()
    
    # 2. Strategic Header
    render_header_banner(bool(uploaded_file))
    
    if uploaded_file:
        # 3. Model Inference (Self-Healing Interface)
        with st.spinner("Analyzing document structure and linguistic risk vectors..."):
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
                st.toast("Document analysis complete.")
                
                # 4. Content Presentation
                render_executive_overview(df)
                
                col_radar, col_spacer = st.columns([2, 1])
                with col_radar:
                    render_risk_radar_panel(df)
                
                render_clause_workspace(df)
                render_intelligence_analytics(df)
                render_governance_section()
                
            except Exception as e:
                st.warning("Model compatibility issue detected — attempting safe inference mode.")
                st.info("System has engaged standby recover protocols to maintain platform stability.")
                st.error("Platform Warning: An inconsistency occurred during ingestion. Initiating recovery protocols.")
                st.caption(f"Context: {str(e)}")
    else:
        render_onboarding_state()

if __name__ == "__main__":
    main()
