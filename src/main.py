from agent.agent import Retriever, BusinessAgent
from agent.tools import extract_metrics_from_csvs

def format_executive_briefing(output: dict):

    risks = output.get("risk_breakdown", {})
    actions = output.get("recommended_actions", [])
    confidence = output.get("reflection", {}).get("confidence", 0)

    print("\n" + "="*70)
    print("EXECUTIVE PROJECT RISK REPORT")
    print("="*70)

    print("\nPROJECT SUMMARY")
    print("-"*70)
    print(output.get("summary", ""))

    print("\nRISK BREAKDOWN")
    print("-"*70)
    for k, v in risks.items():
        print(f"{k.replace('_',' ').title()}: {v}")

    print("\nSTRATEGIC ACTIONS")
    print("-"*70)
    for i, a in enumerate(actions, 1):
        print(f"{i}. {a}")

    print("\nCONFIDENCE LEVEL")
    print("-"*70)
    print(f"{confidence * 100:.0f}% based on system evaluation")

    print("="*70 + "\n")


def run_demo():
    retriever = Retriever()
    tools = {
        "extract_metrics": extract_metrics_from_csvs
    }

    agent = BusinessAgent(retriever, tools)
    query = "Provide an executive risk assessment and strategic recommendations"
    output = agent.handle(query)

    format_executive_briefing(output)


if __name__ == "__main__":
    run_demo()