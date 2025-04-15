from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
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

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is the population of South Korea?"
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config, # type: ignore
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

_ = input()

events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. How about the population of Japan?"
                ),
            },
        ],
    },
    config, # type: ignore
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

_ = input()

events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. How about the population of China?"
                ),
            },
        ],
    },
    config, # type: ignore
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

_ = input()

# This is a simple example of how to replay a state
to_replay = None
for state in graph.get_state_history(config): # type: ignore
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)

    if len(state.values["messages"]) > 0:
        print(state.values["messages"][-1].pretty_print())

    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # ================================== Ai Message ==================================
        # Tool Calls:
        #   tavily_search_results_json (call_YQAiFP2XMgBTeIZB5KHjfLkG)
        #  Call ID: call_YQAiFP2XMgBTeIZB5KHjfLkG
        #   Args:
        #     query: current population of Japan 2023
        # None
        to_replay = state

_ = input()

print("Replay this state *********")
print(to_replay.next)
print(to_replay.config)

# The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
