from dotenv import load_dotenv
load_dotenv()

import os

from langchain_mistralai import ChatMistralAI

print("MISTRAL KEY LOADED:", bool(os.getenv("MISTRAL_API_KEY")))

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=os.getenv("MISTRAL_API_KEY")
)

response = llm.invoke("Say hello in one sentence.")

print("MISTRAL RESPONSE:")
print(response.content)