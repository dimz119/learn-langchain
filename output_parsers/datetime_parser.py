# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate

output_parser = DatetimeOutputParser()
# print(output_parser.get_format_instructions())
"""
Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.

Examples: 0845-05-01T23:51:26.179657Z, 0953-10-03T20:38:54.550930Z, 0188-02-08T08:37:30.449473Z

Return ONLY this string, no other words!
"""
template = """Answer the users question:

{question}

{format_instructions}"""
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)
chain = LLMChain(prompt=prompt, llm=OpenAI())
output_dict = chain.invoke("around when was bitcoin founded?")
print(output_parser.parse(output_dict['text']))
"""
2009-01-03 18:15:05
"""