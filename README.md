# Business Briefing Agent — Capstone (Python)

Summary
-------
Autonomous Python agent that produces executive briefings, detects risks, and proposes prioritized actions from project documentation and CSVs. Project built as the final Capstone for the :contentReference[oaicite:1]{index=1} engineering track.

Key features
------------
- Document ingestion (md/txt) under `src/data/docs/`.
- Metric extraction from CSVs in `src/data/csvs/`.
- Embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`) and local retrieval with FAISS.
- Agent loop: retrieval → planning → tool-calls → reflection.
- Tools available: `extract_metrics`, `create_actions`.
- Simple automatic evaluation using `src/eval/evaluate.py`.

Requirements
------------
- Python 3.10+ (use virtualenv)
- Install dependencies with `pip install -r requirements.txt`.

Quick install
-------------
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt