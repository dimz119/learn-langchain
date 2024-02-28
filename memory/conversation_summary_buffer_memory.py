# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8
from langchain_openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory

llm = OpenAI()

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})

print(memory.load_memory_variables({}))
"""
{'history': "System: \nThe human greets the AI and the AI responds asking what's up.\nHuman: not much you\nAI: not much"}
"""

memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=10, return_messages=True
)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})

print(memory.load_memory_variables({}))
"""
{'history': [
    SystemMessage(content="\nThe human greets the AI. The AI responds by asking what's up."),
    HumanMessage(content='not much you'),
    AIMessage(content='not much')]}
"""

messages = memory.chat_memory.messages
previous_summary = ""
print(memory.predict_new_summary(messages, previous_summary))
"""
The human and AI exchange pleasantries and express that there is not much going on.
"""
