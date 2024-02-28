# Note as of 02/27/2024
# before you start you need to install the following
# pip install langchain==0.1.9 langchain-openai==0.0.8
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that extract the {column} given the data `{data}`"
human_template = "What is the age of {name}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template), # role
    ("human", human_template), # content
])

loader = CSVLoader(file_path='./csv_sample.csv')
data = loader.load()

text_list = []
for record in data:
    text_list.append(record.page_content)

chat_prompt_output = chat_prompt.format_messages(
                        column="age",
                        data=("\n".join(text_list)),
                        name="Minh Barrett")
print(chat_prompt_output)

chat_model = ChatOpenAI()
print(chat_model.invoke(chat_prompt_output))
