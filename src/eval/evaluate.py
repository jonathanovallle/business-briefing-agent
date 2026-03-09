import yaml
import json
from agent.agent import Retriever, BusinessAgent
from agent.tools import extract_metrics_from_csvs, generate_dynamic_actions

def load_tests(path="src/eval/tests.yaml"):
    return yaml.safe_load(open(path, "r", encoding="utf-8"))

def score_output(output, expect_keywords):
    txt = json.dumps(output).lower()
    found = sum(1 for k in expect_keywords if k.lower() in txt)
    return found / max(1, len(expect_keywords))

def main():
    tests = load_tests()
    retriever = Retriever()
    tools = {"extract_metrics": extract_metrics_from_csvs, "create_actions": generate_dynamic_actions}
    agent = BusinessAgent(retriever, tools, use_llm_local=False)  # evaluate without LLM by default
    results = []
    for t in tests:
        out = agent.handle(t['prompt'])
        score = score_output(out, t.get('expect_keywords', []))
        results.append({"id": t['id'], "score": score, "out": out})
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()