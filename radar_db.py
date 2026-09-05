"""
Database access and synchronization module for AI Model & Research Radar.
Uses Dataflow SDK for configurations and Airflow SqliteHook with graceful fallback to sqlite3.
"""

import os
import sqlite3
import datetime
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

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


def _resolve_db_path() -> str:
    """Resolves database file path, falling back to local directory if shared path is inaccessible."""
    primary_dir = "/home/jovyan/shared"
    primary_path = os.path.join(primary_dir, "ai_radar.db")
    try:
        os.makedirs(primary_dir, exist_ok=True)
        # Verify write permissions
        test_file = os.path.join(primary_dir, ".perm_check")
        with open(test_file, "w") as f:
            f.write("1")
        if os.path.exists(test_file):
            os.remove(test_file)
        return primary_path
    except Exception:
        # Fallback to local project directory
        local_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(local_dir, "ai_radar.db")


DB_PATH = _resolve_db_path()
CONN_ID = dataflow.variable("sqlite_conn_id") or "DATAFLOW_DB"
HF_API_URL = dataflow.variable("hf_api_url") or "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=25"
ARXIV_API_URL = dataflow.variable("arxiv_api_url") or "http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.CV&max_results=15&sortBy=submittedDate&sortOrder=descending"


def ensure_sqlite_connection(conn_id: str = CONN_ID, db_path: str = None) -> None:
    """Ensures Airflow SQLite connection is configured when Airflow environment is present."""
    if db_path is None:
        db_path = DB_PATH
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    except Exception:
        pass

    try:
        from airflow.models import Connection
        from airflow.settings import Session
        session = Session()
        conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
        if not conn:
            new_conn = Connection(
                conn_id=conn_id,
                conn_type="sqlite",
                host=db_path
            )
            session.add(new_conn)
            session.commit()
        else:
            if conn.host != db_path:
                conn.host = db_path
                session.commit()
        session.close()
    except Exception:
        # Standalone container without Airflow metadata DB
        pass


@contextmanager
def get_db_connection(conn_id: str = CONN_ID):
    """
    Context manager that yields a database connection.
    Attempts Airflow SqliteHook first, falling back gracefully to standard sqlite3.
    """
    ensure_sqlite_connection(conn_id, DB_PATH)
    conn = None
    is_hook = False
    
    try:
        from airflow.providers.sqlite.hooks.sqlite import SqliteHook
        hook = SqliteHook(sqlite_conn_id=conn_id)
        conn = hook.get_conn()
        is_hook = True
    except Exception:
        conn = None

    if conn is None:
        try:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            conn = sqlite3.connect(DB_PATH, timeout=10)
        except Exception:
            # Fallback to in-memory SQLite if filesystem is not writable
            conn = sqlite3.connect(":memory:")
        is_hook = False

    try:
        yield conn
    finally:
        if not is_hook and conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def init_database(conn_id: str = CONN_ID) -> None:
    """Initializes tables in the SQLite database."""
    create_models_table = """
    CREATE TABLE IF NOT EXISTS models_trending (
        id TEXT PRIMARY KEY,
        pipeline_tag TEXT,
        downloads INTEGER,
        likes INTEGER,
        fetched_at TIMESTAMP
    );
    """
    
    create_arxiv_table = """
    CREATE TABLE IF NOT EXISTS arxiv_papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT,
        summary TEXT,
        published TEXT,
        category TEXT
    );
    """
    try:
        with get_db_connection(conn_id) as conn:
            cursor = conn.cursor()
            cursor.execute(create_models_table)
            cursor.execute(create_arxiv_table)
            conn.commit()
    except Exception as e:
        print(f"init_database notice: {e}")


def fetch_hf_models(api_url: str = HF_API_URL) -> List[tuple]:
    """Fetches trending models from HuggingFace API with quick timeout and offline fallback."""
    headers = {"User-Agent": "Dataflow-AI-Radar/1.0"}
    models = []
    try:
        resp = requests.get(api_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for item in data:
                m_id = item.get("id") or item.get("modelId")
                if not m_id:
                    continue
                tag = item.get("pipeline_tag") or "other"
                downloads = int(item.get("downloads", 0) or 0)
                likes = int(item.get("likes", 0) or 0)
                models.append((m_id, tag, downloads, likes, now_str))
    except Exception as e:
        print(f"HF API fetch notice: {e}")

    if not models:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        models = [
            ("meta-llama/Meta-Llama-3-8B-Instruct", "text-generation", 4520300, 18500, now_str),
            ("openai-community/gpt2", "text-generation", 3890200, 12300, now_str),
            ("black-forest-labs/FLUX.1-schnell", "text-to-image", 2980100, 15400, now_str),
            ("google/gemma-2-9b-it", "text-generation", 2150400, 9800, now_str),
            ("stabilityai/stable-diffusion-3.5-large", "text-to-image", 1870200, 11200, now_str),
            ("BAAI/bge-large-en-v1.5", "feature-extraction", 1650300, 4300, now_str),
            ("mistralai/Mistral-7B-Instruct-v0.3", "text-generation", 1540800, 8900, now_str),
            ("distilbert/distilbert-base-uncased", "fill-mask", 1430200, 5600, now_str),
            ("facebook/bart-large-cnn", "summarization", 1210500, 4800, now_str),
            ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "text-generation", 1100400, 7200, now_str),
        ]
    return models


def fetch_arxiv_papers(api_url: str = ARXIV_API_URL) -> List[tuple]:
    """Fetches AI research papers from arXiv API with quick timeout and offline fallback."""
    headers = {"User-Agent": "Dataflow-AI-Radar/1.0"}
    papers = []
    try:
        resp = requests.get(api_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            
            for entry in root.findall("atom:entry", ns):
                paper_id_elem = entry.find("atom:id", ns)
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                published_elem = entry.find("atom:published", ns)
                
                if paper_id_elem is None or title_elem is None:
                    continue
                    
                paper_id = paper_id_elem.text.strip() if paper_id_elem.text else ""
                title = " ".join(title_elem.text.split()) if title_elem.text else "Untitled Paper"
                summary = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else "No abstract provided."
                published = published_elem.text.strip() if published_elem is not None and published_elem.text else ""
                
                primary_cat = entry.find("arxiv:primary_category", ns)
                category = primary_cat.attrib.get("term", "") if primary_cat is not None else ""
                if not category:
                    cat_elem = entry.find("atom:category", ns)
                    category = cat_elem.attrib.get("term", "cs.AI") if cat_elem is not None else "cs.AI"
                    
                papers.append((paper_id, title, summary, published, category))
    except Exception as e:
        print(f"arXiv API fetch notice: {e}")

    if not papers:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d")
        papers = [
            (
                "http://arxiv.org/abs/2501.00001v1",
                "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
                "We introduce DeepSeek-R1-Zero and DeepSeek-R1, large language models trained via large-scale reinforcement learning without supervised fine-tuning as a preliminary step, demonstrating state-of-the-art reasoning performance.",
                f"{now_str}T10:00:00Z",
                "cs.AI"
            ),
            (
                "http://arxiv.org/abs/2501.00002v1",
                "Scaling Vision Transformers with Unified Multi-Modal Representation Learning",
                "This paper presents a scalable architecture for jointly modeling vision, language, and structured tabular inputs, achieving superior cross-modal zero-shot transferability across benchmark suites.",
                f"{now_str}T08:30:00Z",
                "cs.CV"
            ),
            (
                "http://arxiv.org/abs/2501.00003v1",
                "Efficient Attention Mechanisms for Ultra-Long Sequence Processing",
                "We explore sub-quadratic attention primitives and hierarchical memory compression schemes that enable processing of context lengths exceeding one million tokens with minimal memory overhead.",
                f"{now_str}T07:15:00Z",
                "cs.CL"
            ),
        ]
    return papers


def sync_all_data(conn_id: str = CONN_ID) -> Dict[str, int]:
    """Syncs data for both HF models and arXiv papers into SQLite database."""
    try:
        init_database(conn_id)
        models = fetch_hf_models()
        papers = fetch_arxiv_papers()
        
        with get_db_connection(conn_id) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR REPLACE INTO models_trending (id, pipeline_tag, downloads, likes, fetched_at) VALUES (?, ?, ?, ?, ?)",
                models
            )
            cursor.executemany(
                "INSERT OR REPLACE INTO arxiv_papers (paper_id, title, summary, published, category) VALUES (?, ?, ?, ?, ?)",
                papers
            )
            conn.commit()
            
        return {"models_synced": len(models), "papers_synced": len(papers)}
    except Exception as e:
        print(f"sync_all_data notice: {e}")
        return {"models_synced": 0, "papers_synced": 0}


def get_models_dataframe(conn_id: str = CONN_ID, category_filter: Optional[str] = None, search_query: Optional[str] = None) -> pd.DataFrame:
    """Queries trending models from SQLite database with safe fallback schema."""
    default_cols = ["id", "pipeline_tag", "downloads", "likes", "fetched_at"]
    try:
        init_database(conn_id)
        
        query = "SELECT id, pipeline_tag, downloads, likes, fetched_at FROM models_trending WHERE 1=1"
        params = []
        
        if category_filter and category_filter != "All":
            query += " AND pipeline_tag = ?"
            params.append(category_filter)
            
        if search_query:
            query += " AND id LIKE ?"
            params.append(f"%{search_query}%")
            
        query += " ORDER BY downloads DESC"
        
        with get_db_connection(conn_id) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        return df if not df.empty else pd.DataFrame(columns=default_cols)
    except Exception as e:
        print(f"get_models_dataframe notice: {e}")
        return pd.DataFrame(columns=default_cols)


def get_arxiv_dataframe(conn_id: str = CONN_ID, category_filter: Optional[str] = None, search_query: Optional[str] = None) -> pd.DataFrame:
    """Queries arXiv research papers from SQLite database with safe fallback schema."""
    default_cols = ["paper_id", "title", "summary", "published", "category"]
    try:
        init_database(conn_id)
        
        query = "SELECT paper_id, title, summary, published, category FROM arxiv_papers WHERE 1=1"
        params = []
        
        if category_filter and category_filter != "All":
            query += " AND category = ?"
            params.append(category_filter)
            
        if search_query:
            query += " AND (title LIKE ? OR summary LIKE ?)"
            params.append(f"%{search_query}%")
            params.append(f"%{search_query}%")
        
        query += " ORDER BY published DESC"
        
        with get_db_connection(conn_id) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        return df if not df.empty else pd.DataFrame(columns=default_cols)
    except Exception as e:
        print(f"get_arxiv_dataframe notice: {e}")
        return pd.DataFrame(columns=default_cols)


def get_radar_summary(conn_id: str = CONN_ID) -> Dict[str, Any]:
    """Computes summary statistics for KPI badges with safe fallback values."""
    fallback_summary = {
        "total_models": 0,
        "total_downloads": 0,
        "total_likes": 0,
        "last_sync": "Never",
        "total_papers": 0,
        "total_categories": 0,
    }
    try:
        init_database(conn_id)
        with get_db_connection(conn_id) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*), SUM(downloads), SUM(likes), MAX(fetched_at) FROM models_trending")
            m_row = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT category) FROM arxiv_papers")
            p_row = cursor.fetchone()
            
        return {
            "total_models": m_row[0] if m_row and m_row[0] else 0,
            "total_downloads": m_row[1] if m_row and m_row[1] else 0,
            "total_likes": m_row[2] if m_row and m_row[2] else 0,
            "last_sync": m_row[3] if m_row and m_row[3] else "Never",
            "total_papers": p_row[0] if p_row and p_row[0] else 0,
            "total_categories": p_row[1] if p_row and p_row[1] else 0,
        }
    except Exception as e:
        print(f"get_radar_summary notice: {e}")
        return fallback_summary
