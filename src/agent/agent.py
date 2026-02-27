# agent.py
import json, os, subprocess
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

META_PATH = "src/data/faiss_meta.json"
INDEX_PATH = "src/data/faiss_index.bin"
EMBED_MODEL = "all-MiniLM-L6-v2"

@dataclass
class ToolResult:
    name: str
    success: bool
    output: str

class Retriever:
    def __init__(self):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.ids = meta["ids"]
        self.index = faiss.read_index(INDEX_PATH)
        self.model = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, k:int=3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        results = []
        for score_idx in I[0]:
            if score_idx < len(self.ids):
                fname = self.ids[score_idx]
                path = f"src/data/docs/{fname}"
                with open(path, "r", encoding="utf-8") as f:
                    results.append({"file": fname, "text": f.read()})
        return results, D[0].tolist()  # texts and distances

class BusinessAgent:
    def __init__(self, retriever, tools: Dict[str, callable], use_llm_local: bool=False):
        self.retriever = retriever
        self.tools = tools
        self.history = []
        self.use_llm_local = use_llm_local

    def plan(self, query, context_texts):
        # simple plan: extract bullets + metrics + actions
        prompt = f"Key points: {query}\n\nContext:\n{context_texts[:4000]}"
        if self.use_llm_local and "llm_run" in self.tools:
            out = self.tools["llm_run"](prompt)
            return out
        # fallback rule based summary
        lines = context_texts.splitlines()
        top = [l.strip() for l in lines if l.strip()][:6]
        return "Resumen (reglas):\n" + "\n".join(top)

    def reflect(self, retrieval_scores, tool_logs):
 # robust reflection: prefer tool success rate as confidence
        try:
            avg_score = float(sum(retrieval_scores) / max(1, len(retrieval_scores)))
            # ignore absurd distances
            if avg_score > 1e6 or avg_score != avg_score:  # >1e6 o NaN
                avg_score_norm = None
            else:
                avg_score_norm = avg_score
        except Exception:
            avg_score_norm = None

        success_rate = sum(1 for t in tool_logs if t.success) / max(1, len(tool_logs))

        # confidence estable basado en éxito de las tools
        confidence = round(success_rate, 2)

        return {
            "avg_distance": avg_score_norm,
            "tool_success": success_rate,
            "confidence": confidence
        }

    def handle(self, query: str):
        contexts, scores = self.retriever.retrieve(query, k=4)
        context_text = "\n\n".join([f"{c['file']}:\n{c['text']}" for c in contexts])
        plan = self.plan(query, context_text)

        logs = []
        # call extract_metrics
        if "extract_metrics" in self.tools:
            try:
                res = self.tools["extract_metrics"]()
                logs.append(ToolResult("extract_metrics", True, res))
            except Exception as e:
                logs.append(ToolResult("extract_metrics", False, str(e)))
        # create summary or actions
        if "create_actions" in self.tools:
            try:
                res = self.tools["create_actions"](plan + "\n\n" + context_text)
                logs.append(ToolResult("create_actions", True, res))
            except Exception as e:
                logs.append(ToolResult("create_actions", False, str(e)))

        reflection = self.reflect(scores, logs)
        result = {
            "query": query,
            "plan": plan,
            "tool_logs": [t.__dict__ for t in logs],
            "reflection": reflection
        }
        self.history.append(result)
        return result