from langsmith import Client
from openai import OpenAI
from langchain_core.messages import convert_to_openai_messages

# Connect to LangSmith and OpenAI

client = Client()
oai_client = OpenAI()

# Pull the prompt to use

# You can also specify a specific commit by passing the commit hash "my-prompt:<commit-hash>"

prompt = client.pull_prompt("my-fistprompt")

# Since our prompt 
# only has one variable we could also pass in the value directly

# The code below is equivalent to formatted_prompt = prompt.invoke("What is the color of the sky?")

formatted_prompt = prompt.invoke({"question": "What is the color of the sky?"})

# Test the prompt

response = oai_client.chat.completions.create(
    model="gpt-4o",
    messages=convert_to_openai_messages(formatted_prompt.messages), # type: ignore
)
print(response)
