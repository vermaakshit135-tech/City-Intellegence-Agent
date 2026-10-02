from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool

from langchain_core.messages import HumanMessage

from rich import print




@tool
def get_text_length(text : str) -> int:
    """Calculating the length of a given text."""
    return len(text)

llm = ChatMistralAI(model = "ministral-3b-latest")

akshit = llm.bind_tools([get_text_length])

message = []
prompt = input("you : ")
query = HumanMessage(content=prompt)

message.append(query)


result = akshit.invoke(message)


message.append(result)
print(message)