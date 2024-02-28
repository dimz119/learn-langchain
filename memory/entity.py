# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8
from langchain_openai import OpenAI
from langchain.memory import ConversationEntityMemory

llm = OpenAI(temperature=0)
memory = ConversationEntityMemory(llm=llm)
_input = {"input": "Deven & Sam are working on a hackathon project"}
memory.load_memory_variables(_input)
memory.save_context(
    _input,
    {"output": " That sounds like a great project! What kind of project are they working on?"}
)

print(memory.load_memory_variables({"input": 'who is Sam'}))
"""
{'history': 'Human: Deven & Sam are working on a hackathon project\n
    AI:  That sounds like a great project! What kind of project are they working on?', 
'entities': {'Sam': 'Sam is working on a hackathon project with Deven.'}}
"""

print(memory.load_memory_variables({"input": 'who is Deven'}))
"""
{'history': 'Human: Deven & Sam are working on a hackathon project\n
    AI:  That sounds like a great project! What kind of project are they working on?', 
'entities': {
    'Deven': 'Deven is working on a hackathon project with Sam.',
    'Sam': 'Sam is working on a hackathon project with Deven.'}}
"""