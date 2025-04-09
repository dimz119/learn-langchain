from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

# Connect to the LangSmith client

client = Client()

# Define the prompt

prompt = ChatPromptTemplate([
    ("system", "You are a helpful chatbot. Answer the question as best as you can. provide the answer within 1 line"),
    ("user", "{question}"),
])

# Push the prompt
client.push_prompt("my-fistprompt", object=prompt)
