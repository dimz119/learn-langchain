from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})

print(memory.load_memory_variables({}))
"""
{'history': 'Human: not much you\nAI: not much'}
"""

memory = ConversationBufferWindowMemory(k=1, return_messages=True)
memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})

print(memory.load_memory_variables({}))
"""
{'history': [HumanMessage(content='not much you'), AIMessage(content='not much')]}
"""

