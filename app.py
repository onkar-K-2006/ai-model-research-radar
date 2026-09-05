"""
Streamlit Application: AI Model & Research Radar
Interactive dashboard for monitoring trending Hugging Face AI models and the latest arXiv research papers.
Connects via Airflow's SqliteHook to the Dataflow SQLite database.
"""

import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Safe Dataflow SDK import with fallback guard
try:
    from dataflow.dataflow import Dataflow
    dataflow = Dataflow()
except Exception:
    class _FallbackDataflow:
        def variable(self, name: str):
            return None
        def secret(self, name: str):
            return None
        def variable_or_secret(self, key: str):
            return None
    dataflow = _FallbackDataflow()

# Safe database utilities import
try:
    from radar_db import (
        CONN_ID,
        init_database,
        sync_all_data,
        get_models_dataframe,
        get_arxiv_dataframe,
        get_radar_summary,
    )
except ImportError:
    # Ensure current directory is in sys.path if invoked from another working directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from radar_db import (
        CONN_ID,
        init_database,
        sync_all_data,
        get_models_dataframe,
        get_arxiv_dataframe,
        get_radar_summary,
    )

# Page Configuration
st.set_page_config(
    page_title="AI Model & Research Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2F6 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .paper-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 6px;
    }
    .badge-tag {
        background-color: #E0E7FF;
        color: #3730A3;
    }
    .badge-cat {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .badge-date {
        background-color: #F1F5F9;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)


# Initialize and seed database if empty
@st.cache_resource
def setup_app_db():
    try:
        init_database(CONN_ID)
        summary = get_radar_summary(CONN_ID)
        if summary.get("total_models", 0) == 0 or summary.get("total_papers", 0) == 0:
            sync_all_data(CONN_ID)
        return True
    except Exception as e:
        return False

setup_app_db()

# Sidebar Controls
with st.sidebar:
    st.markdown("### 📡 Radar Controls")
    st.caption(f"Connection ID: `{CONN_ID}`")
    
    st.markdown("---")
    st.markdown("#### 🔄 Data Pipeline Sync")
    if st.button("🚀 Run Live Data Ingestion", use_container_width=True, type="primary"):
        with st.spinner("Fetching latest Hugging Face models & arXiv papers..."):
            res = sync_all_data(CONN_ID)
            st.success(f"Synced {res.get('models_synced', 0)} models & {res.get('papers_synced', 0)} papers!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### ℹ️ About AI Radar")
    st.info(
        "**AI Model & Research Radar** aggregates top trending open-source AI weights from "
        "HuggingFace and cutting-edge research publications from arXiv (cs.AI, cs.CL, cs.CV). "
        "Automated hourly via Airflow DAG `ai_model_paper_radar_dag`."
    )
    
    summary_data = get_radar_summary(CONN_ID)
    st.markdown(f"**Last Database Update:**\n`{summary_data.get('last_sync', 'Never')}`")

# Header section
st.markdown("<div class='main-header'>📡 AI Model & Research Radar</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Real-time intelligence dashboard tracking trending machine learning models and groundbreaking arXiv AI research papers.</div>",
    unsafe_allow_html=True
)

# KPI Summary Metrics Bar
summary = get_radar_summary(CONN_ID)
col1, col2, col3, col4 = st.columns(4)

total_models = summary.get("total_models", 0)
total_downloads = summary.get("total_downloads", 0)
total_likes = summary.get("total_likes", 0)
total_papers = summary.get("total_papers", 0)

with col1:
    st.metric(label="🔥 Tracked Top Models", value=f"{total_models:,}")

with col2:
    st.metric(label="📥 Total Model Downloads", value=f"{total_downloads / 1_000_000:.2f}M" if total_downloads > 0 else "0")

with col3:
    st.metric(label="❤️ Total Model Likes", value=f"{total_likes:,}")

with col4:
    st.metric(label="📜 Curated arXiv Papers", value=f"{total_papers:,}")

st.markdown("---")

# Main Navigation Tabs
tab1, tab2 = st.tabs([
    "🔥 Trending Models Leaderboard", 
    "📜 arXiv Paper Summarizer"
])

# ==============================================================================
# TAB 1: Trending Models Leaderboard
# ==============================================================================
with tab1:
    st.markdown("### 🔥 Top Trending HuggingFace Models")
    st.caption("Live models ranked by aggregate downloads, pipeline category, and community likes.")
    
    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
    
    # Fetch all models for filter options
    df_all_models = get_models_dataframe(CONN_ID)
    available_tags = ["All"] + sorted(list(df_all_models["pipeline_tag"].dropna().unique())) if not df_all_models.empty else ["All"]
    
    with f_col1:
        selected_category = st.selectbox(
            "Filter by Pipeline Tag / Category:",
            options=available_tags,
            index=0,
            key="model_cat_filter"
        )
        
    with f_col2:
        search_model = st.text_input(
            "Search Model Name / Author:",
            placeholder="e.g. llama, gpt2, flux, mistral...",
            key="model_search"
        )
        
    with f_col3:
        top_n = st.selectbox("Show Top N:", options=[10, 15, 25, 50], index=1)

    # Filtered Data
    df_filtered = get_models_dataframe(CONN_ID, category_filter=selected_category, search_query=search_model)
    
    if df_filtered.empty:
        st.warning("No models found matching the specified filters.")
    else:
        df_top = df_filtered.head(top_n)
        
        # Horizontal Bar Chart (Plotly)
        st.markdown("#### 📊 Downloads Distribution Leaderboard")
        
        # Prepare chart dataframe (sorted ascending for top-to-bottom bar chart)
        chart_df = df_top.sort_values(by="downloads", ascending=True).copy()
        chart_df["downloads_formatted"] = chart_df["downloads"].apply(lambda x: f"{x:,}")
        
        fig = px.bar(
            chart_df,
            x="downloads",
            y="id",
            orientation="h",
            color="pipeline_tag",
            text="downloads_formatted",
            title=f"Top {len(chart_df)} Most Downloaded Models ({selected_category})",
            labels={"downloads": "Total Downloads", "id": "Model ID", "pipeline_tag": "Pipeline Tag"},
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=max(450, len(chart_df) * 32),
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(248, 250, 252, 0.6)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Downloads Count", gridcolor="#E2E8F0"),
            yaxis=dict(title="", tickfont=dict(size=12)),
            margin=dict(l=10, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            marker=dict(line=dict(width=1, color="#334155"))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive Table with formatted columns and HuggingFace links
        st.markdown("#### 📋 Detailed Models Catalog")
        
        display_df = df_filtered.copy()
        display_df["HuggingFace Link"] = display_df["id"].apply(lambda m_id: f"https://huggingface.co/{m_id}")
        
        # Format metrics
        display_df["Downloads"] = display_df["downloads"].apply(lambda x: f"{x:,}")
        display_df["Likes"] = display_df["likes"].apply(lambda x: f"{x:,}")
        display_df["Pipeline Tag"] = display_df["pipeline_tag"]
        display_df["Last Fetched"] = display_df["fetched_at"]
        display_df["Model ID"] = display_df["id"]
        
        st.dataframe(
            display_df[["Model ID", "Pipeline Tag", "Downloads", "Likes", "HuggingFace Link", "Last Fetched"]],
            column_config={
                "HuggingFace Link": st.column_config.LinkColumn(
                    "Model Repository",
                    display_text="Open on Hugging Face ↗"
                ),
                "Model ID": st.column_config.TextColumn("Model Identifier", width="medium"),
                "Downloads": st.column_config.TextColumn("Downloads 📥", width="small"),
                "Likes": st.column_config.TextColumn("Likes ❤️", width="small"),
                "Pipeline Tag": st.column_config.TextColumn("Category Tag", width="small"),
            },
            use_container_width=True,
            hide_index=True,
        )


# ==============================================================================
# TAB 2: arXiv Paper Summarizer
# ==============================================================================
with tab2:
    st.markdown("### 📜 arXiv Research Paper Summarizer")
    st.caption("Latest preprints from Computer Science (cs.AI, cs.CL, cs.CV) with concise executive summaries.")
    
    # Paper Filters
    p_col1, p_col2 = st.columns([1, 2])
    
    df_all_papers = get_arxiv_dataframe(CONN_ID)
    available_cats = ["All"] + sorted(list(df_all_papers["category"].dropna().unique())) if not df_all_papers.empty else ["All"]
    
    with p_col1:
        selected_paper_cat = st.selectbox(
            "Filter by arXiv Category:",
            options=available_cats,
            index=0,
            key="arxiv_cat_filter"
        )
        
    with p_col2:
        search_paper = st.text_input(
            "Search Title or Abstract Keywords:",
            placeholder="e.g. reasoning, diffusion, transformer, alignment, multi-modal...",
            key="arxiv_search"
        )
        
    df_papers_filtered = get_arxiv_dataframe(CONN_ID, category_filter=selected_paper_cat, search_query=search_paper)
    
    if df_papers_filtered.empty:
        st.info("No arXiv papers found matching your query.")
    else:
        st.markdown(f"Showing **{len(df_papers_filtered)}** research papers:")
        
        for idx, row in df_papers_filtered.iterrows():
            paper_id = row["paper_id"]
            title = row["title"]
            summary = row["summary"]
            published = str(row["published"])[:10]
            category = row["category"]
            
            # Construct PDF URL from paper_id
            pdf_url = paper_id.replace("/abs/", "/pdf/")
            if not pdf_url.endswith(".pdf") and "/pdf/" in pdf_url:
                pdf_url += ".pdf"
                
            with st.expander(f"📌 [{category}] {title} ({published})", expanded=(idx == 0)):
                st.markdown(
                    f"<span class='badge badge-cat'>🏷️ {category}</span>"
                    f"<span class='badge badge-date'>📅 Published: {published}</span>",
                    unsafe_allow_html=True
                )
                
                st.markdown("#### 📖 Executive Summary / Abstract:")
                st.markdown(f"> {summary}")
                
                st.markdown("---")
                link_col1, link_col2, _ = st.columns([1.5, 1.5, 5])
                with link_col1:
                    st.link_button("📄 View on arXiv", url=paper_id, use_container_width=True)
                with link_col2:
                    st.link_button("📥 Download PDF", url=pdf_url, use_container_width=True)
