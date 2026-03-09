import json
import argparse
from agent.agent import Retriever, BusinessAgent
from agent.tools import extract_metrics_from_csvs, llm_run_ollama
from typing import Any, Dict

def print_human_briefing(out: Dict[str, Any]) -> None:
    print("\n" + "="*70)
    print("EXECUTIVE PROJECT BRIEFING")
    print("="*70 + "\n")

    # Summary / briefing_text can be either a string or a list of bullets
    briefing = out.get("briefing")
    print("SUMMARY:")
    print("-"*70)
    if isinstance(briefing, list):
        for b in briefing:
            print("• " + b)
    else:
        # could be an auto-fallback short string
        print(str(briefing))
    print("\n")

    # Show llm_plan if present (evidence of agentic decision)
    llm_plan = out.get("llm_plan")
    if llm_plan:
        print("LLM PLAN (evidence of tool-calls):")
        print("-"*70)
        try:
            print(json.dumps(llm_plan, indent=2, ensure_ascii=False))
        except Exception:
            print(str(llm_plan))
        print("\n")

    # Risks
    risks = out.get("risk_breakdown", {})
    print("RISK BREAKDOWN:")
    print("-"*70)
    if risks:
        for k, v in risks.items():
            print(f"- {k.replace('_',' ').title()}: {v}")
    else:
        print("- No risks detected or analysis missing")
    print("\n")

    # Actions
    actions = out.get("recommended_actions", [])
    print("RECOMMENDED ACTIONS:")
    print("-"*70)
    if actions:
        for i, a in enumerate(actions, start=1):
            print(f"{i}. {a}")
    else:
        print("- No actions generated")
    print("\n")

    # Confidence
    conf = out.get("reflection", {}).get("confidence")
    if conf is not None:
        try:
            pct = float(conf) * 100
            print(f"CONFIDENCE: {pct:.0f}% (computed by reflection module)")
        except Exception:
            print(f"CONFIDENCE: {conf}")
    else:
        print("CONFIDENCE: not available")
    print("\n" + "="*70 + "\n")

def run_demo(use_llm: bool = True, query: str = None, timeout: int = 120, as_json: bool = False, save_path: str = None):
    retriever = Retriever()
    tools = {"extract_metrics": extract_metrics_from_csvs}
    if use_llm:
        # llm_run wrapper (model name 'mistral' used as example)
        tools["llm_run"] = lambda prompt: llm_run_ollama(prompt, model="mistral", timeout=timeout)

    agent = BusinessAgent(retriever, tools, use_llm_local=use_llm)

    if not query:
        query = "Provide an executive risk assessment and strategic recommendations"

    output = agent.handle(query)

    if as_json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print_human_briefing(output)
        # print a compact JSON footer so mentors can see keys quickly
        print("DEBUG: Keys in result ->", ", ".join(output.keys()))

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Result saved to {save_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Run Business Briefing Agent demo")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM usage (use rule-based fallback)")
    p.add_argument("--query", type=str, default=None, help="Query to send to agent (wrap in quotes)")
    p.add_argument("--json", action="store_true", dest="as_json", help="Print full JSON output instead of human-friendly briefing")
    p.add_argument("--timeout", type=int, default=120, help="LLM timeout in seconds (if using local Ollama)")
    p.add_argument("--save", type=str, default=None, help="Save JSON output to this file path")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_demo(use_llm=not args.no_llm, query=args.query, timeout=args.timeout, as_json=args.as_json, save_path=args.save)