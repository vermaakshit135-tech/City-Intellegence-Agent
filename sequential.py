from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableLambda

short_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in short way"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in detailed way"
)



model = ChatMistralAI(model = "ministral-3b-latest")

parser = StrOutputParser()

chain = RunnableParallel({
    "short" :RunnableLambda(lambda x : x ['short'] ) | short_prompt | model | parser , 
    "detailed" :RunnableLambda(lambda x : x ['detailed']) | detailed_prompt | model | parser

})

result = chain.invoke({
    "short" : {"topic" : "machine learning"},
    "detailed" : {"topic" : "deep learning"}
})
print(result['short'])
print(result['detailed'])
