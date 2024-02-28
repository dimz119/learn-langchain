# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8 
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt_output = prompt.format(product="colorful socks")
print(prompt_output)
# What is a good name for a company that makes colorful socks?

from langchain.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template), # role
    ("human", human_template), # content
])

chat_prompt_output = chat_prompt.format_messages(
                        input_language="English",
                        output_language="French",
                        text="I love programming.")
print(chat_prompt_output)
# [SystemMessage(content='You are a helpful assistant that translates English to French.'),
#   HumanMessage(content='I love programming.')]


from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI()
print(chat_model.invoke(chat_prompt_output))
