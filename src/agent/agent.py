import json
import re
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
    def __init__(self, retriever: Retriever, tools: Dict[str, Any], use_llm_local: bool = False):
        self.retriever = retriever
        self.tools = tools
        self.use_llm_local = use_llm_local
        self.history: List[Dict[str, Any]] = []

    def plan(self, query: str, context_texts: str) -> str:
        # fallback summary (rule-based)
        lines = (context_texts or "").splitlines()
        top = [l.strip() for l in lines if l.strip()][:6]
        return "Summary (rule-based):\n" + "\n".join(top)

    def tools_manifest_brief(self):
        # return simple manifest text for LLM prompt
        try:
            from agent.tools import TOOLS_MANIFEST
            return {k: v["description"] for k, v in TOOLS_MANIFEST.items()}
        except Exception:
            return {}

    def reflect(self, retrieval_scores: List[float], tool_logs: List[ToolResult],
                risk_breakdown: Dict[str, str], metrics: Dict[str, Any]) -> Dict[str, Any]:
        try:
            avg_score = float(sum(retrieval_scores) / max(1, len(retrieval_scores)))
            if avg_score > 1e6 or avg_score != avg_score:
                avg_score_norm = None
            else:
                avg_score_norm = avg_score
        except Exception:
            avg_score_norm = None

        success_rate = sum(1 for t in tool_logs if t.success) / max(1, len(tool_logs))

        high_risks = sum(1 for v in (risk_breakdown or {}).values() if str(v).lower() == "high")
        risk_penalty = min(0.3, 0.1 * high_risks)

        if not metrics:
            data_penalty = 0.2
        elif len(metrics) < 3:
            data_penalty = 0.1
        else:
            data_penalty = 0.0

        raw_confidence = success_rate - risk_penalty - data_penalty
        confidence = max(0.0, min(0.99, raw_confidence))
        confidence = round(confidence, 2)

        return {
            "avg_distance": avg_score_norm,
            "tool_success": round(success_rate, 2),
            "confidence": confidence
        }

    def handle(self, query: str) -> Dict[str, Any]:
        # 1) Retrieval (limit contexts to top-3 to keep LLM prompt small)
        contexts, scores = self.retriever.retrieve(query, k=6)
        limited_contexts = contexts[:3]
        context_text = "\n\n".join([f"{c['file']}:\n{c['text']}" for c in limited_contexts])
        MAX_CHARS = 2000
        if len(context_text) > MAX_CHARS:
            context_text = context_text[:MAX_CHARS] + "\n\n[context truncated]"

        # 2) Ask LLM (if available) for a plan: summary_request + tool_calls
        manifest = self.tools_manifest_brief()
        system_msg = (
            "You are an assistant that MUST respond ONLY with valid JSON and nothing else. "
            "Return a JSON object with keys: "
            "'summary_request' (string), "
            "'tool_calls' (array of objects with 'tool' and 'args'), "
            "and 'final_instructions' (string). "
            "Do NOT include any extra commentary or markdown. "
            "If you cannot answer, return an empty 'tool_calls' array. "
            f"\n\nAvailable tools: {json.dumps(manifest)}"
        )
        user_msg = f"User query: {query}\n\nContext:\n{context_text}\n\nRespond with the JSON plan."

        prompt_text = "SYSTEM:\n" + system_msg + "\n\nUSER:\n" + user_msg

        llm_plan = None
        use_llm = self.use_llm_local and "llm_run" in self.tools

        if use_llm:
            try:
                raw = self.tools["llm_run"](prompt_text)
                # extract JSON object from raw response
                m = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
                json_text = m.group(1) if m else raw
                llm_plan = json.loads(json_text)
            except Exception as e:
                # fallback: simple sequential plan
                llm_plan = {
                    "summary_request": "Summarize key points",
                    "tool_calls": [
                        {"tool": "extract_metrics", "args": {}},
                        {"tool": "analyze_risks", "args": {"context_text": context_text}},
                        {"tool": "generate_dynamic_actions", "args": {"risks": {}}}
                    ],
                    "final_instructions": "Produce briefing and actions."
                }
        else:
            llm_plan = {
                "summary_request": "Summarize key points",
                "tool_calls": [
                    {"tool": "extract_metrics", "args": {}},
                    {"tool": "analyze_risks", "args": {"context_text": context_text}},
                    {"tool": "generate_dynamic_actions", "args": {"risks": {}}}
                ],
                "final_instructions": "Produce briefing and actions."
            }

        # 3) Execute tool calls (LLM-decided)
        tool_logs: List[ToolResult] = []
        metrics_data = {}
        risks = {}
        actions: List[str] = []

        for call in llm_plan.get("tool_calls", []):
            tool_name = call.get("tool")
            args = call.get("args", {}) or {}
            try:
                if tool_name == "extract_metrics":
                    res = self.tools["extract_metrics"]()
                    metrics_data = res.get("metrics", {}) if isinstance(res, dict) else {}
                    tool_logs.append(ToolResult(tool_name, True, str(res)))

                elif tool_name == "analyze_risks":
                    from agent.tools import analyze_risks
                    ctx = args.get("context_text", context_text)
                    risks = analyze_risks(ctx, metrics_data)
                    tool_logs.append(ToolResult(tool_name, True, str(risks)))

                elif tool_name == "generate_dynamic_actions":
                    from agent.tools import generate_dynamic_actions
                    r = args.get("risks", risks)
                    if isinstance(r, str):
                        if isinstance(risks, dict) and r in risks:
                            r = {r: risks[r]}
                        else:
                            r = risks
                    if isinstance(r, list):
                        r = {k: risks.get(k, "Unknown") for k in r}
                    actions = generate_dynamic_actions(r)
                    tool_logs.append(ToolResult(tool_name, True, "\n".join(actions)))

                else:
                    tool_logs.append(ToolResult(tool_name, False, f"Unknown tool {tool_name}"))

            except Exception as e:
                tool_logs.append(ToolResult(tool_name, False, str(e)))

        # 4) Finalize with LLM: produce executive briefing + self-evaluation
        final_briefing = None
        self_evaluation = None
        if use_llm:
            try:
                final_system = (
                    "You are an expert business analyst. You MUST respond ONLY with valid JSON. "
                    "Return a JSON object with keys: "
                    "'briefing_text' (string) and "
                    "'self_evaluation' (string). "
                    "The 'briefing_text' should be a concise executive summary (3-6 short bullets or 2-3 short paragraphs) "
                    "including key insights, top risks and 2-3 prioritized actions. "
                    "The 'self_evaluation' should be 2–3 short sentences about limitations and what additional data would improve confidence. "
                    "Do NOT include any additional explanation outside the JSON."
                )
                final_user = (
                    f"Query: {query}\n\nContext:\n{context_text}\n\n"
                    f"Metrics: {json.dumps(metrics_data)}\n\n"
                    f"Risks: {json.dumps(risks)}\n\n"
                    f"Actions: {json.dumps(actions)}\n\n"
                    "Return JSON."
                )
                raw2 = self.tools["llm_run"](final_system + "\n\n" + final_user)
                m2 = re.search(r"(\{.*\})", raw2, flags=re.DOTALL)
                jt = m2.group(1) if m2 else raw2
                parsed = json.loads(jt)
                final_briefing = parsed.get("briefing_text")
                self_evaluation = parsed.get("self_evaluation")
            except Exception as e:
                final_briefing = None
                self_evaluation = f"LLM finalization failed: {str(e)}"

        if not final_briefing:
            # fallback basic composition
            final_briefing = "Summary (auto):\n" + (context_text.splitlines()[:6] and "\n".join(context_text.splitlines()[:6]) or "")
            self_evaluation = "Self-evaluation: LLM not available for finalization; basic fallback used."

        # 5) Reflection
        reflection = self.reflect(scores, tool_logs, risks, metrics_data)

        try:
            result_plan = llm_plan if llm_plan is not None else {}
        except Exception:
            result_plan = {}

        result = {
            "query": query,
            "llm_plan": result_plan,
            "briefing": final_briefing,
            "self_evaluation": self_evaluation,
            "risk_breakdown": risks,
            "recommended_actions": actions,
            "metrics": metrics_data,
            "tool_logs": [t.__dict__ for t in tool_logs],
            "reflection": reflection
        }
        self.history.append(result)
        return result