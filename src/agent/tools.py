import os, csv

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

            if rows and "value" in rows[0]:
                total = 0.0
                for r in rows:
                    try:
                        total += float(r.get("value", 0))
                    except:
                        pass
                metrics[f"{fname}_total_value"] = total

    return {"metrics": metrics, "summary": "Metrics extracted successfully"}


def analyze_risks(context_text: str, metrics: dict):
    context_lower = context_text.lower()

    financial_risk = "Low"
    operational_risk = "Low"
    data_risk = "Low"

    # Financial risk logic
    if "marketing spend increased" in context_lower and "revenue increased" not in context_lower:
        financial_risk = "High"
    elif "marketing spend increased" in context_lower:
        financial_risk = "Medium"

    # Operational risk logic
    if "churn increased" in context_lower:
        operational_risk = "High"
    elif "churn decreased" in context_lower:
        operational_risk = "Low"

    # Data risk logic
    if not metrics:
        data_risk = "High"
    elif len(metrics) < 2:
        data_risk = "Medium"

    return {
        "financial_risk": financial_risk,
        "operational_risk": operational_risk,
        "data_risk": data_risk
    }


def generate_dynamic_actions(risks: dict):
    actions = []

    if risks["financial_risk"] in ["High", "Medium"]:
        actions.append("Conduct marketing ROI analysis and cost efficiency review.")

    if risks["operational_risk"] == "High":
        actions.append("Launch immediate customer retention and churn mitigation strategy.")

    if risks["data_risk"] in ["High", "Medium"]:
        actions.append("Implement data completeness audit and improve reporting visibility.")

    if not actions:
        actions.append("Maintain current strategy and monitor KPIs weekly.")

    return actions