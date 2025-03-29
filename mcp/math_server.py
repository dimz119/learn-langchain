# Imports FastMCP, a lightweight framework for exposing Python functions as MCP-compatible tools.
# This lets other programs (like agents) call your Python functions over a standard interface (like stdio or HTTP).
from mcp.server.fastmcp import FastMCP

# Creates an instance of FastMCP and names the tool "Math".
# This name is optional, but it's helpful for debugging or when managing multiple tools.
mcp = FastMCP("Math")

# The @mcp.tool() decorator exposes this function as a callable tool via MCP.
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    # If this script is run directly, it starts the MCP server using stdio (standard input/output).
    # This allows the agent (in your first script) to connect to this tool by launching it as a subprocess
    mcp.run(transport="stdio")
