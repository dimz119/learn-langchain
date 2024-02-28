# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

# LangChain Expression Language(LCEL)
chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()
print(chain.invoke({"text": "colors"}))
# ['red', 'blue', 'green', 'yellow', 'purple']
