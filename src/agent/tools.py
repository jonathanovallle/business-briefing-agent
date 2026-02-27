# tools.py
import os, csv, io, json, subprocess, sys

# If using Ollama locally, implement llm_run
def llm_run_ollama(prompt: str):
    # calls: ollama run mistral "<prompt>"
    proc = subprocess.run(["ollama", "run", "mistral", prompt], capture_output=True, text=True)
    if proc.returncode != 0:
        return proc.stderr or "LLM error"
    return proc.stdout

def extract_metrics_from_csvs(csv_dir="src/data/csvs"):
    # aggregate simple metrics from CSVs
    out = []
    for fname in os.listdir(csv_dir):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            out.append(f"{fname}: rows={len(rows)}")
            # example: if 'rev' column exists, sum it
            if rows and 'rev' in rows[0]:
                total = 0.0
                for r in rows:
                    try:
                        total += float(r.get('rev',0))
                    except:
                        pass
                out.append(f"{fname}: total_rev={total}")
    return "\n".join(out) or "No CSVs or no metrics found"

def create_actions_from_summary(text: str):
    # simple heuristics to create 3 actions
    actions = [
        "1) Reunir datos faltantes y actualizar dashboard en 3 días.",
        "2) Priorizar top 2 riesgos con owner asignado.",
        "3) Abrir issue de seguimiento y plan de mitigación."
    ]
    return "\n".join(actions)