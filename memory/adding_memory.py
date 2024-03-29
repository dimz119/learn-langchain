# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)

inputs = {"input": "Hello, My name is Joon"}
response = chain.invoke(inputs)
print(response)
"""
content="Hello Joon! It's nice to meet you. How can I assist you today?"
"""
memory.save_context(inputs, {"output": response.content}) # type: ignore

print(memory.load_memory_variables({}))
"""
{'history': [HumanMessage(content='Hello, My name is Joon'), AIMessage(content='Hello Joon! How can I assist you today?')]}
"""

inputs = {"input": "whats my name"}
response = chain.invoke(inputs)
print(response)
"""
content='Your name is Joon. How can I assist you further, Joon?
"""
