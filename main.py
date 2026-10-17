from langgraph.graph import END, START, StateGraph
from typing import TypedDict,Dict
from langchain_core.documents import Document
from RAG import filter_new_news, find_similar_news, storing
from Scraper import scrapeNews
from Agent import *
import json
class AgentState(TypedDict):
    News: list[Document]
    NewsToStore: list[Document]
    History: list[dict]
    Summary:str
    Encyclopedia: Dict

def scrape_node(state: AgentState) -> AgentState:
    state["News"] = scrapeNews()
    state["NewsToStore"] = filter_new_news(state["News"])
    return state

def agent_node(state: AgentState)-> AgentState:
    state["Summary"] = run_agent(state["News"])
    return state

def history_decision_node(state: AgentState) -> str:
    """Route to historical context only when RAG finds a similar article."""
    matches = find_similar_news(state["NewsToStore"])
    state["History"] = matches
    return "check_history" if matches else "skip_history"

def history_node(state: AgentState) -> AgentState:
    return state

def skip_history_node(state: AgentState) -> AgentState:
    return state

def store_node(state: AgentState) -> AgentState:
    storing(state["NewsToStore"])
    return state

def vuln_node(state: AgentState)-> AgentState:
    try:
        state["Encyclopedia"] = Encyclopedia(state["Summary"])
    except ValueError as error:
        print(f"Warning: could not parse vulnerabilities: {error}")
        state["Encyclopedia"] = {"vulnerabilities": []}
    return state

# Creating the Graph

Graph = StateGraph(AgentState)
Graph.add_node("scrape_news", scrape_node)
Graph.add_node("history_decision", history_decision_node)
Graph.add_node("check_history", history_node)
Graph.add_node("skip_history", skip_history_node)
Graph.add_node("analyze_news", agent_node)
Graph.add_node("store_news", store_node)
Graph.add_node("vuln_news",vuln_node)
Graph.add_edge(START, "scrape_news")
Graph.add_conditional_edges(
    "scrape_news",
    history_decision_node,
    {
        "check_history": "check_history",
        "skip_history": "skip_history",
    },
)
Graph.add_edge("check_history", "analyze_news")
Graph.add_edge("skip_history", "analyze_news")
Graph.add_edge("analyze_news","vuln_news")
Graph.add_edge("vuln_news", "store_news")
Graph.add_edge("store_news", END)
app = Graph.compile()


if __name__ == "__main__":
    result = app.invoke({
        "News": [],
        "NewsToStore": [],
        "History": [],
        "Summary": "",
        "Encyclopedia": {},
    })
    print(result["Summary"])
    vul = result["Encyclopedia"]
    try:
        with open("Encyclopedia.json", "r") as f:
            encyclopedia = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        encyclopedia = {"vulnerabilities": []}

    encyclopedia.setdefault("vulnerabilities", []).extend(
        vul.get("vulnerabilities", [])
    )
    with open("Encyclopedia.json", "a") as f:
        json.dump(encyclopedia, f, indent=4)
