# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8 
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
"""
Your response should be a list of comma separated values, eg: `foo, bar, baz`
"""
# check what to expect to LLM
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0)

_input = prompt.format(subject="ice cream flavors")
output = model.invoke(_input)

print(output_parser.parse(output))
# 모델 버전이 바뀐 후로는 
# ['1. Chocolate\n2. Vanilla\n3. Strawberry\n4. Mint chocolate chip\n5. Cookies and cream']
# 으로 아웃풋이 됩니다.
