# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
prompt_value = prompt.invoke({"topic": "ice cream"})
"""
messages=[HumanMessage(content='tell me a short joke about ice cream')]
"""
print(prompt_value.to_string())
"""
Human: tell me a short joke about ice cream
"""

model = ChatOpenAI()
message = model.invoke(prompt_value)
"""
content='Why did the ice cream go to therapy? \n\nBecause it had too many scoops of emotions!'
"""

# from langchain.llms import OpenAI
# llm = OpenAI(model="gpt-3.5-turbo-instruct")
# llm_message = llm.invoke(prompt_value)
"""
Why did the ice cream go to therapy?

Because it was having a meltdown!
"""

output_parser = StrOutputParser()
print(output_parser.invoke(message))
"""
Why did the ice cream go to therapy?

Because it had a meltdown!
"""

# similar to unix pipe operator
chain = prompt | model | output_parser

result = chain.invoke({"topic": "ice cream"})
print(result)
"""
Why did the ice cream go to therapy?

Because it had too many sprinkles of anxiety!
# """
