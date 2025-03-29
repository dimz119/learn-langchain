# Imports Python's built-in asyncio module, which is used to run asynchronous code using async/await.
import asyncio

# ClientSession: manages a session with an MCP-compliant tool or service.
# StdioServerParameters: used to specify how to start the tool (like a subprocess).
from mcp import ClientSession, StdioServerParameters

# connects to a tool over standard input/output.
from mcp.client.stdio import stdio_client

# Imports a helper function that loads tools that support the MCP protocol, making them compatible with LangChain.
from langchain_mcp_adapters.tools import load_mcp_tools

# Imports a function to create an agent that follows the ReAct pattern (Reasoning + Acting) from LangGraph.
# ReAct agents can use tools and think step-by-step.
from langgraph.prebuilt import create_react_agent

# Imports the OpenAI chat model interface from LangChain.
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

# Prepares the parameters to launch a tool server via math_server.py using Python.
server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["/Users/seungjoonlee/git/learn-langchain/mcp/math_server.py"],
)

# Defines an asynchronous main function.
async def main():
    # Starts the math_server.py subprocess and establishes a connection using stdin and stdout.
    # Returns read and write streams for communication.
    async with stdio_client(server_params) as (read, write):
        # Wraps the read/write streams in a ClientSession to interact with the tool more easily.
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Loads the tools exposed by the running math_server.py.
            # These tools are now in a format that the LangChain agent can use.
            tools = await load_mcp_tools(session)

            # Creates an agent that can reason and call the MCP tools using the GPT-4o model.
            agent = create_react_agent(model, tools)
            
            # Invokes the agent asynchronously and gives it a user message.
            # The agent will process the question, possibly call the math tool, and return a response.
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(agent_response)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
