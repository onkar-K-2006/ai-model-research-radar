# 📡 AI Model & Research Radar

An intelligence radar dashboard for tracking trending Hugging Face AI models and the latest arXiv research publications in Computer Science (`cs.AI`, `cs.CL`, `cs.CV`).

## Architecture & Dataflow Integration

- **Database**: SQLite storage located at `/home/jovyan/shared/ai_radar.db` with tables:
  - `models_trending`: `(id, pipeline_tag, downloads, likes, fetched_at)`
  - `arxiv_papers`: `(paper_id, title, summary, published, category)`
- **Airflow DAG**: `ai_model_paper_radar_dag` running hourly at `0 * * * *` using `SqliteHook(sqlite_conn_id='DATAFLOW_DB')` to ingest and upsert records.
- **Streamlit Dashboard**:
  - **Tab 1: 🔥 Trending Models Leaderboard**: Real-time ranking with category filters and interactive horizontal Plotly bar chart.
  - **Tab 2: 📜 arXiv Paper Summarizer**: Expandable research cards with abstract summaries, categorization badges, and direct arXiv PDF links.
- **Dataflow SDK**: Uses `Dataflow()` to dynamically configure `sqlite_conn_id`, `hf_api_url`, and `arxiv_api_url`.

## Configuration Keys

| Key | Type | Default Value | Description |
|---|---|---|---|
| `sqlite_conn_id` | Variable | `DATAFLOW_DB` | Airflow Connection ID for SQLite |
| `hf_api_url` | Variable | `https://huggingface.co/api/models?sort=downloads&direction=-1&limit=25` | Hugging Face Trending Models API |
| `arxiv_api_url` | Variable | `http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.CV&max_results=15&sortBy=submittedDate&sortOrder=descending` | arXiv Atom API Feed |
