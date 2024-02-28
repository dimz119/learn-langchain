# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8
from typing import List
from langchain_openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

model_name = "gpt-3.5-turbo-instruct"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

# Here's another example, but with a compound typed field.
class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

print(parser.get_format_instructions())
"""
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"name": {"title": "Name", "description": "name of an actor", "type": "string"}, "film_names": {"title": "Film Names", "description": "list of names of films they starred in", "type": "array", "items": {"type": "string"}}}, "required": ["name", "film_names"]}
```
"""
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=actor_query)

output = model.invoke(_input.to_string())
"""
Output:
{"name": "Tom Hanks", "film_names": ["Forrest Gump", "Saving Private Ryan", "Cast Away", "The Green Mile", "Apollo 13", "Toy Story", "Toy Story 2", "Toy Story 3", "The Da Vinci Code", "Catch Me If You Can"]}
name='Tom Hanks' film_names=['Forrest Gump', 'Saving Private Ryan', 'Cast Away', 'The Green Mile', 'Apollo 13', 'Toy Story', 'Toy Story 2', 'Toy Story 3', 'The Da Vinci Code', 'Catch Me If You Can']
"""
m = parser.parse(output)
print(m)
"""
name='Tom Hanks' film_names=['Forrest Gump', 'Saving Private Ryan', 'The Green Mile', 'Cast Away', 'Toy Story']
"""
print(m.name)