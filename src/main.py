# main.py
from agent.agent import Retriever, BusinessAgent
from agent.tools import extract_metrics_from_csvs, create_actions_from_summary, llm_run_ollama

def format_briefing(output: dict):
    reflection = output.get("reflection", {})
    tool_logs = output.get("tool_logs", [])
    
    confidence = reflection.get("confidence", 0)
    
    if confidence >= 0.8:
        health = "🟢 Low Risk"
    elif confidence >= 0.5:
        health = "🟡 Moderate Risk"
    else:
        health = "🔴 High Risk"
    
    print("\n" + "="*60)
    print("EXECUTIVE PROJECT BRIEFING")
    print("="*60)
    print(f"\nProject Health: {health}")
    
    print("\nKey Summary:")
    print("-"*60)
    print(output.get("plan", "No summary available"))
    
    print("\nRecommended Actions:")
    print("-"*60)
    for log in tool_logs:
        if log["name"] == "create_actions" and log["success"]:
            print(log["output"])
    
    print("\nConfidence Level:")
    print("-"*60)
    print(f"{confidence * 100:.0f}% based on tool execution success")
    print("="*60 + "\n")

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
    
    format_briefing(out)

if __name__ == "__main__":
    run_demo(use_llm=False)