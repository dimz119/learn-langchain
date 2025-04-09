from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm = ChatOpenAI(model="gpt-4o") # type: ignore
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# Add a tool node to the graph
# This node will be called when the tool is invoked
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile with memory
graph = graph_builder.compile(checkpointer=memory)
# print(graph.get_graph().draw_mermaid())
# https://mermaid.live/

config = {"configurable": {"thread_id": "1"}}

user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config, # type: ignore
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

user_input = "Remember my name?"

config = {"configurable": {"thread_id": "2"}}

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config, # type: ignore
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

# events = graph.stream(
#     {"messages": [{"role": "user", "content": user_input}]},
#     {"configurable": {"thread_id": "2"}},
#     stream_mode="values",
# )
# for event in events:
#     event["messages"][-1].pretty_print()

# config = {"configurable": {"thread_id": "2"}}

snapshot = graph.get_state(config) # type: ignore
print(snapshot)
# refer to chatbot_with_memory.json
# print(snapshot)