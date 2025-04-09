from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("What is the population of South Korean?") # type: ignore

