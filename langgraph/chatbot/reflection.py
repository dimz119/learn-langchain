from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 3-paragraph essays."
            " Generate the best essay possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm = ChatOpenAI(model="gpt-4o") # type: ignore
generate = prompt | llm

essay = ""
request = HumanMessage(
    content="Write an essay on why the little prince is relevant in modern childhood"
)
for chunk in generate.stream({"messages": [request]}):
    print(chunk.content, end="")
    essay += chunk.content

# Start reflecting on the essay
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
            " Provide detailed recommendations, including requests for length, depth, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm
reflection = ""
for chunk in reflect.stream({"messages": [request, HumanMessage(content=essay)]}):
    print(chunk.content, end="")
    reflection += chunk.content

# import asyncio
# from typing import Annotated, List, Sequence
# from langgraph.graph import END, StateGraph, START
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver
# from typing_extensions import TypedDict


# class State(TypedDict):
#     messages: Annotated[list, add_messages]


# async def generation_node(state: State) -> State:
#     return {"messages": [await generate.ainvoke(state["messages"])]} # type: ignore


# async def reflection_node(state: State) -> State:
#     # Other messages we need to adjust
#     cls_map = {
#         "ai": HumanMessage,
#         "human": AIMessage
#     }
#     # First message is the original user request. We hold it the same for all nodes
#     translated = [state["messages"][0]] + [
#         cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
#     ]
#     res = await reflect.ainvoke(translated)
#     # We treat the output of this as human feedback for the generator
#     return {"messages": [HumanMessage(content=res.content)]}


# builder = StateGraph(State)
# builder.add_node("generate", generation_node)
# builder.add_node("reflect", reflection_node)
# builder.add_edge(START, "generate")


# def should_continue(state: State):
#     if len(state["messages"]) > 6:
#         # End after 3 iterations
#         return END
#     return "reflect"


# builder.add_conditional_edges("generate", should_continue)
# builder.add_edge("reflect", "generate")
# memory = MemorySaver()
# graph = builder.compile(checkpointer=memory)
# # print(graph.get_graph().draw_mermaid())

# config = {"configurable": {"thread_id": "1"}}

# async def main():
#     async for event in graph.astream(
#         {
#             "messages": [
#                 HumanMessage(
#                     content="Generate an essay on the topicality of The Little Prince and its message in modern life"
#                 )
#             ],
#         },
#         config,
#     ):
#         print(event)
#         print("---")

# if __name__ == "__main__":
#     asyncio.run(main())