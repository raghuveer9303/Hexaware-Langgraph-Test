import os
import tkinter as tk
from tkinter import filedialog

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prompt import SYSTEM_PROMPT

load_dotenv()

folder = os.path.dirname(__file__)
budget_file = None

print("Welcome to your Budget Financial Planner!")
print("Select a CSV file to analyze spending, or cancel to continue without one.")
print()

root = tk.Tk()
root.withdraw()
chosen = filedialog.askopenfilename(
    title="Select budget CSV",
    filetypes=[("CSV files", "*.csv")],
    initialdir=folder,
)
root.destroy()

if chosen and os.path.isfile(chosen):
    budget_file = chosen
else:
    print("No CSV loaded")
print()


@tool
def read_budget_csv():
    """Read the budget CSV file with expenses."""
    if not budget_file or not os.path.isfile(budget_file):
        return "No budget CSV is loaded."
    with open(budget_file) as f:
        return f.read()


_graph_ref: dict = {}


@tool
def save_architecture_graph():
    """Save a PNG diagram of this agent's architecture (nodes, tools, and flow).

    Use when the user asks to visualize the graph, see the architecture, workflow,
    or how the agent or tools work.
    """
    compiled = _graph_ref.get("graph")
    if compiled is None:
        return "Graph is not available yet."
    graph_png = os.path.join(os.getcwd(), "graph.png")
    with open(graph_png, "wb") as f:
        f.write(compiled.get_graph(xray=True).draw_mermaid_png())
    return f"Graph saved to {graph_png}"


tools = [read_budget_csv, save_architecture_graph]
model = ChatMistralAI(model="mistral-small")
model = model.bind_tools(tools)


def call_model(state):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    reply = model.invoke(messages)
    return {"messages": reply}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)
_graph_ref["graph"] = graph

print("Budget Financial Planner")
print("Type quit to exit")
print()

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        print("Bye!")
        break
    if not user_input:
        continue

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    )

    answer = result["messages"][-1].content
    print("Bot:", answer)
    print()
