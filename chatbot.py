import os
import sqlite3
import uuid
import tkinter as tk
from tkinter import filedialog

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite import SqliteSaver
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


def _normalize_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def stream_chat(user_input, config):
    printed_bot_prefix = False
    in_assistant_line = False
    tool_started = set()

    for mode, chunk in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            message, metadata = chunk
            if metadata.get("langgraph_node") != "call_model":
                continue

            if message.tool_call_chunks:
                for tool_call in message.tool_call_chunks:
                    tool_name = tool_call.get("name") or "unknown_tool"
                    tool_id = tool_call.get("id") or tool_name
                    if tool_id not in tool_started:
                        if in_assistant_line:
                            print()
                            in_assistant_line = False
                        print(f"[Tool call] {tool_name}")
                        tool_started.add(tool_id)

            text = _normalize_content(message.content)
            if text:
                if not printed_bot_prefix:
                    print("Bot: ", end="", flush=True)
                    printed_bot_prefix = True
                print(text, end="", flush=True)
                in_assistant_line = True

        elif mode == "updates":
            tools_update = chunk.get("tools")
            if tools_update and in_assistant_line:
                print()
                in_assistant_line = False

            if tools_update:
                for item in tools_update.get("messages", []):
                    name = getattr(item, "name", None)
                    if name:
                        print(f"[Tool result] {name}")

    if in_assistant_line:
        print()
    if not printed_bot_prefix:
        print("Bot: ")
    print()


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)
builder.add_edge("tools", "call_model")

db_path = os.path.join(folder, "checkpoints.sqlite")
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)
memory.setup()

graph = builder.compile(checkpointer=memory)
_graph_ref["graph"] = graph


def choose_thread_id():
    if env_id := os.environ.get("THREAD_ID"):
        return env_id

    print("Conversation thread")
    print("  1) New conversation")
    print("  2) Resume existing thread")
    while True:
        choice = input("Choose (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("Please enter 1 or 2.")

    if choice == "1":
        thread_id = str(uuid.uuid4())
        print()
        print(f"Your thread ID (save this to resume later): {thread_id}")
        print()
        return thread_id

    while True:
        thread_id = input("Enter your thread ID: ").strip()
        if thread_id:
            break
        print("Thread ID cannot be empty.")

    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    n = len(snapshot.values.get("messages", [])) if snapshot else 0
    print()
    if n:
        print(f"Resuming thread {thread_id} ({n} messages in history)")
    else:
        print(f"No prior messages for thread {thread_id}. Starting fresh.")
    print()
    return thread_id


thread_id = choose_thread_id()
config = {"configurable": {"thread_id": thread_id}}

print("Budget Financial Planner")
print("Type quit to exit")
print()

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        print("Bye!")
        print(f"Thread ID: {thread_id}")
        break
    if not user_input:
        continue

    stream_chat(user_input, config)
