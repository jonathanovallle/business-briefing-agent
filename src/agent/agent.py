import json
from typing import Dict, Any, List
from dataclasses import dataclass
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

    def retrieve(self, query: str, k: int = 3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        # Normalize or not depending on how index was built. Keep raw for now.
        D, I = self.index.search(q_emb, k)
        results = []
        scores = []
        for idx, score in zip(I[0], D[0].tolist()):
            if idx < len(self.ids):
                fname = self.ids[idx]
                path = f"src/data/docs/{fname}"
                with open(path, "r", encoding="utf-8") as f:
                    results.append({"file": fname, "text": f.read()})
                scores.append(float(score))
        return results, scores

class BusinessAgent:
    def __init__(self, retriever: Retriever, tools: Dict[str, callable], use_llm_local: bool = False):
        self.retriever = retriever
        self.tools = tools
        self.history: List[Dict[str, Any]] = []
        self.use_llm_local = use_llm_local

    def plan(self, query: str, context_texts: str) -> str:
        prompt = f"Key points: {query}\n\nContext:\n{context_texts[:4000]}"
        if self.use_llm_local and "llm_run" in self.tools:
            return self.tools["llm_run"](prompt)
        lines = context_texts.splitlines()
        top = [l.strip() for l in lines if l.strip()][:6]
        return "Summary (rule-based):\n" + "\n".join(top)

    def reflect(self, retrieval_scores: List[float], tool_logs: List[ToolResult],
                risk_breakdown: Dict[str, str], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute a robust reflection object that includes:
          - avg_distance: normalized retrieval score or None if absurd
          - tool_success: fraction of tools that executed successfully
          - confidence: realistic confidence based on tool success, detected risks and available metrics
        The confidence formula:
          confidence = success_rate - risk_penalty - data_penalty
          where risk_penalty = min(0.3, 0.1 * number_of_high_risks)
                data_penalty = 0.2 if no metrics, 0.1 if few metrics (<3)
          final confidence clamped to [0.0, 0.99]
        """
        # avg retrieval score (may be distance or similarity depending on index)
        try:
            avg_score = float(sum(retrieval_scores) / max(1, len(retrieval_scores)))
            if avg_score > 1e6 or avg_score != avg_score:  # ignore absurd values or NaN
                avg_score_norm = None
            else:
                avg_score_norm = avg_score
        except Exception:
            avg_score_norm = None

        # tool success rate
        success_rate = sum(1 for t in tool_logs if t.success) / max(1, len(tool_logs))

        # risk penalty: each "High" risk reduces confidence by 0.1, capped at 0.3
        high_risks = sum(1 for v in risk_breakdown.values() if str(v).lower() == "high")
        risk_penalty = min(0.3, 0.1 * high_risks)

        # data penalty: no metrics => -0.2, few metrics (<3) => -0.1
        if not metrics:
            data_penalty = 0.2
        elif len(metrics) < 3:
            data_penalty = 0.1
        else:
            data_penalty = 0.0

        raw_confidence = success_rate - risk_penalty - data_penalty
        # clamp and avoid 1.0 (never show 100%)
        confidence = max(0.0, min(0.99, raw_confidence))
        confidence = round(confidence, 2)

        return {
            "avg_distance": avg_score_norm,
            "tool_success": round(success_rate, 2),
            "confidence": confidence
        }

    def handle(self, query: str) -> Dict[str, Any]:
        contexts, scores = self.retriever.retrieve(query, k=4)
        context_text = "\n\n".join([f"{c['file']}:\n{c['text']}" for c in contexts])

        plan = self.plan(query, context_text)

        logs: List[ToolResult] = []
        metrics_data: Dict[str, Any] = {}

        # Extract metrics (tool reads CSVs from disk)
        if "extract_metrics" in self.tools:
            try:
                res = self.tools["extract_metrics"]()
                # tool returns dict {"metrics": {...}, "summary": "..."}
                if isinstance(res, dict) and "metrics" in res:
                    metrics_data = res.get("metrics", {})
                logs.append(ToolResult("extract_metrics", True, str(res)))
            except Exception as e:
                logs.append(ToolResult("extract_metrics", False, str(e)))

        # Risk analysis and dynamic actions (tools implemented in tools.py)
        from agent.tools import analyze_risks, generate_dynamic_actions

        risks = analyze_risks(context_text, metrics_data)
        actions = generate_dynamic_actions(risks)

        logs.append(ToolResult("risk_analysis", True, str(risks)))
        logs.append(ToolResult("dynamic_actions", True, "\n".join(actions)))

        # compute reflection/confidence using tools results, risks and metrics
        reflection = self.reflect(scores, logs, risks, metrics_data)

        result = {
            "query": query,
            "summary": plan,
            "risk_breakdown": risks,
            "recommended_actions": actions,
            "metrics": metrics_data,
            "reflection": reflection
        }

        self.history.append(result)
        return result