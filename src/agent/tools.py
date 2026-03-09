import os
import csv
import subprocess
import json

def llm_run_ollama(prompt: str, model: str = "mistral", timeout: int = 120):
    try:
        proc = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or "LLM returned non-zero exit")
        return proc.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"LLM timeout after {timeout}s")
    except Exception as e:
        raise RuntimeError(str(e))

def extract_metrics_from_csvs(csv_dir="src/data/csvs"):
    metrics = {}
    if not os.path.exists(csv_dir):
        return {"metrics": {}, "summary": "No CSV files found"}

    for fname in os.listdir(csv_dir):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            metrics[f"{fname}_rows"] = len(rows)
            # if there is a numeric column 'value' or 'rev' sum it
            if rows:
                # try common numeric columns
                for col in ("value", "rev", "revenue", "amount"):
                    if col in rows[0]:
                        total = 0.0
                        for r in rows:
                            try:
                                total += float(r.get(col, 0))
                            except:
                                pass
                        metrics[f"{fname}_total_{col}"] = total
                        break
    return {"metrics": metrics, "summary": "Metrics extracted successfully"}

def analyze_risks(context_text: str, metrics: dict):
    context_lower = (context_text or "").lower()
    financial_risk = "Low"
    operational_risk = "Low"
    data_risk = "Low"

    # Financial heuristic
    if "marketing spend increased" in context_lower and "revenue increased" not in context_lower:
        financial_risk = "High"
    elif "marketing spend increased" in context_lower:
        financial_risk = "Medium"

    # Operational heuristic
    if "churn increased" in context_lower or "backlog" in context_lower and "issues" in context_lower:
        operational_risk = "High"
    elif "churn decreased" in context_lower:
        operational_risk = "Low"

    # Data heuristic
    if not metrics:
        data_risk = "High"
    elif len(metrics) < 3:
        data_risk = "Medium"

    return {
        "financial_risk": financial_risk,
        "operational_risk": operational_risk,
        "data_risk": data_risk
    }

def generate_dynamic_actions(risks: dict):
    actions = []
    if not isinstance(risks, dict):
        risks = risks or {}

    fin = risks.get("financial_risk", "Low")
    op = risks.get("operational_risk", "Low")
    data = risks.get("data_risk", "Low")

    if fin in ("High", "Medium"):
        actions.append("Conduct a marketing ROI analysis and cost-efficiency review.")
    if op == "High":
        actions.append("Initiate an immediate customer retention and churn mitigation plan.")
    if data in ("High", "Medium"):
        actions.append("Run a data completeness audit and improve reporting visibility.")
    if not actions:
        actions.append("Maintain current strategy and monitor key KPIs weekly.")
    return actions

TOOLS_MANIFEST = {
    "extract_metrics": {
        "description": "Extract aggregate numeric metrics from CSV files stored on disk. No args required."
    },
    "analyze_risks": {
        "description": "Analyze risks from context text and metrics. Args: context_text (string), metrics (json)."
    },
    "generate_dynamic_actions": {
        "description": "Produce prioritized action items based on a risk breakdown. Args: risks (json)."
    }
}