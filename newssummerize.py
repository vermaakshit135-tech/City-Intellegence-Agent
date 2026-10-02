from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


search_tool = TavilySearchResults(max_result = 5)

model = ChatMistralAI(model = "ministral-3b-latest")

prompt = ChatPromptTemplate.from_template(
    """
you are an ai assistance 

you will give an news from the sracth which are real and based on the AI 


{news}

"""
)

chain = prompt | model | StrOutputParser()
news_result = search_tool.run("give the latest AI news")
result = chain.invoke({"news" : news_result})
print(result) 








