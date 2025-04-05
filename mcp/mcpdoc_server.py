# Imports Python's built-in asyncio module, which is used to run asynchronous code using async/await.
import asyncio

# Import MultiServerMCPClient to connect to multiple servers at once.
from langchain_mcp_adapters.client import MultiServerMCPClient

# Imports a function to create an agent that follows the ReAct pattern (Reasoning + Acting) from LangGraph.
# ReAct agents can use tools and think step-by-step.
from langgraph.prebuilt import create_react_agent

# Imports the OpenAI chat model interface from LangChain.
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")

# Defines an asynchronous main function.
async def main():
    async with MultiServerMCPClient(
        {
            "langgraph-docs-mcp": {
                "url": "http://localhost:8082/sse",
                "transport": "sse",
            } # type: ignore
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        langgraph_response = await agent.ainvoke({"messages": "how does langchain chatbot works? check the latest documenation. but make it short as 5 lines"})
        print(langgraph_response)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
