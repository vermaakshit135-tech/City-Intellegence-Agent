from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain.tools import tool
from tavily import TavilyClient
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, ToolMessage


# =========================================================
# WEATHER TOOL
# =========================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of the city."""

    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    data = response.json()

    print("DEBUG:", data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    desc = data["weather"][0]["description"]

    return (
        f"Weather in {city}: "
        f"{temp}°C, feels like {feels_like}°C, "
        f"humidity {humidity}%, {desc}."
    )


# =========================================================
# TAVILY CLIENT
# =========================================================

tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# =========================================================
# NEWS TOOL
# =========================================================

@tool
def get_news(city: str) -> str:
    """Get the latest news of a city."""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return "No news found."

    news_list = []

    for r in results:

        title = r.get("title", "No title")
        url = r.get("url", "")

        news_list.append(
            f"Title: {title}\n"
            f"URL: {url}"
        )

    return "\n\n".join(news_list)


# =========================================================
# TOOLS DICTIONARY
# =========================================================

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}


# =========================================================
# GROQ LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================================================
# BIND TOOLS
# =========================================================

akshit = llm.bind_tools(
    [get_weather, get_news]
)


# =========================================================
# CITY INTELLIGENCE SYSTEM
# =========================================================

messages = []

print("City Intelligence System")
print("Press 0 to exit")


while True:

    prompt = input("you : ")

    if prompt.lower() == "0":
        print("Goodbye!")
        break

    messages.append(
        HumanMessage(content=prompt)
    )


    # =====================================================
    # AGENT LOOP
    # =====================================================

    while True:

        result = akshit.invoke(messages)

        messages.append(result)


        # =================================================
        # CHECK WHETHER MODEL WANTS TO CALL A TOOL
        # =================================================

        if not result.tool_calls:

            print("AI:", result.content)
            break


        # =================================================
        # PROCESS TOOL CALLS
        # =================================================

        for tool_call in result.tool_calls:

            tool_name = tool_call["name"]

            print(
                f"\nAgent wants to call tool: {tool_name}"
            )

            confirm = input(
                "Approve? (y/n): "
            )


            # =============================================
            # DENY TOOL
            # =============================================

            if confirm.lower() != "y":

                print("Tool call was denied.")

                messages.append(
                    ToolMessage(
                        content="The user denied this tool call.",
                        tool_call_id=tool_call["id"]
                    )
                )

                continue


            # =============================================
            # CALL TOOL
            # =============================================

            selected_tool = tools[tool_name]

            tool_result = selected_tool.invoke(
                tool_call["args"]
            )


            print(
                f"\nTool Result:\n{tool_result}"
            )


            # =============================================
            # SEND TOOL RESULT BACK TO MODEL
            # =============================================

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )

        # Go back to LLM so it can generate final answer
        continue