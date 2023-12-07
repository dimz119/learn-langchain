from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

# When LLM returns the response in text format
print(CommaSeparatedListOutputParser().parse("hi, bye"))
# >> ['hi', 'bye']