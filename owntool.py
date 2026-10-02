from langchain.tools import tool


@tool
def get_greeting(name : str) -> str:
    """Generating a greeting message for a user."""
    return f"Hello, {name}! Nice to meet you."


result = get_greeting.invoke({"name" : "akshit"})
print(result)
