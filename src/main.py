# main.py
from agent.agent import Retriever, BusinessAgent
from agent.tools import extract_metrics_from_csvs, create_actions_from_summary, llm_run_ollama
import json

def run_demo(use_llm=False):
    retriever = Retriever()
    tools = {
        "extract_metrics": extract_metrics_from_csvs,
        "create_actions": create_actions_from_summary
    }
    if use_llm:
        tools["llm_run"] = llm_run_ollama

    agent = BusinessAgent(retriever, tools, use_llm_local=use_llm)
    query = "Give me an executive briefing about project risks and recommended actions"
    out = agent.handle(query)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # Cambia a True si tienes Ollama + modelo
    run_demo(use_llm=False)