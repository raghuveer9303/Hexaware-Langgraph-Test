from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, START, StateGraph

from prompt import SYSTEM_PROMPT

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def call_model(state):
    response = model.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

print("LangGraph docs chatbot")
print("Type quit to exit")
print()

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        print("Bye!")
        break
    if user_input == "":
        continue

    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    )

    answer = result["messages"][-1].content
    if isinstance(answer, list):
        answer = "".join(
            part.get("text", "")
            for part in answer
            if isinstance(part, dict) and part.get("type") == "text"
        )

    print("Bot:", answer)
    print()
