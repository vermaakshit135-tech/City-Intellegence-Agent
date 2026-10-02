from dotenv import load_dotenv
load_dotenv()

import os
import json
from html import escape as esc
from urllib.parse import urlparse

import requests
import streamlit as st
from tavily import TavilyClient

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage

MODEL = "openai/gpt-oss-20b"

st.set_page_config(page_title="City Intelligence", page_icon="🌆", layout="centered")


# =========================================================
# TOOLS  (return JSON strings so the UI can draw rich cards)
# =========================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of the city."""
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": os.getenv("OPENWEATHER_API_KEY"), "units": "metric"},
            timeout=10,
        )
        data = r.json()
    except requests.RequestException as e:
        return json.dumps({"error": f"Network error: {e}"})

    if str(data.get("cod")) != "200":
        return json.dumps({"error": data.get("message", "Could not fetch weather")})

    return json.dumps({
        "city": data.get("name", city),
        "country": data.get("sys", {}).get("country", ""),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "wind": data.get("wind", {}).get("speed", 0),
        "condition": data["weather"][0]["main"],
        "description": data["weather"][0]["description"],
    })


@tool
def get_news(city: str) -> str:
    """Get the latest news of a city."""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        res = client.search(query=f"latest news in {city}", search_depth="basic", max_results=3)
    except Exception as e:
        return json.dumps({"error": str(e)})

    items = [{"title": r.get("title", "No title"), "url": r.get("url", "")}
             for r in res.get("results", [])]
    if not items:
        return json.dumps({"error": "No news found."})
    return json.dumps({"city": city, "news": items})


TOOLS = {"get_weather": get_weather, "get_news": get_news}


@st.cache_resource
def get_agent():
    return ChatGroq(model=MODEL, temperature=0).bind_tools([get_weather, get_news])


# =========================================================
# STYLE
# =========================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
:root{--bg:#0b0f1a;--surface:#131a2b;--line:#24304a;--sky:#5cc8ff;--amber:#ffb454;--text:#e8ecf6;--muted:#8b97b3;}
.stApp{font-family:'DM Sans',sans-serif;color:var(--text);
 background:radial-gradient(1000px 420px at 50% -120px,rgba(92,200,255,.16),transparent 70%),var(--bg);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
section[data-testid="stSidebar"]{background:#0e1424;border-right:1px solid var(--line);}

.hero{padding:14px 0 6px;}
.hero h1{font-family:'Space Grotesk',sans-serif;font-size:3rem;font-weight:700;letter-spacing:-.03em;margin:0;color:#fff;}
.hero p{color:var(--muted);margin:6px 0 0;font-size:1.05rem;}

.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin:6px 0;}
.chip{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.8rem;margin:2px 6px 2px 0;border:1px solid var(--line);}
.on{color:#7ee2b8;border-color:rgba(126,226,184,.4);background:rgba(126,226,184,.08);}
.off{color:#ff8a8a;border-color:rgba(255,138,138,.4);background:rgba(255,138,138,.08);}

/* weather: the one loud element */
.wx{display:flex;align-items:center;gap:16px;}
.wx .emo{font-size:3.4rem;line-height:1;}
.wx .city{font-weight:600;font-size:1.15rem;}
.wx .desc{color:var(--muted);text-transform:capitalize;}
.wx .temp{margin-left:auto;font-family:'Space Grotesk',sans-serif;font-size:3.6rem;font-weight:700;letter-spacing:-.04em;color:var(--sky);text-shadow:0 0 32px rgba(92,200,255,.45);}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px;}
.tile{background:#0e1424;border:1px solid var(--line);border-radius:12px;padding:10px 12px;}
.tile span{display:block;color:var(--muted);font-size:.78rem;}
.tile b{font-size:1.1rem;font-weight:600;}

.news{display:block;padding:11px 14px;border-radius:12px;background:#0e1424;border:1px solid var(--line);
 margin-top:8px;text-decoration:none;color:var(--text)!important;transition:border-color .2s,transform .2s;}
.news:hover{border-color:var(--sky);transform:translateX(3px);}
.news small{color:var(--sky);}

.approve{border:1px solid rgba(255,180,84,.55);animation:glow 2.4s ease-in-out infinite;}
.approve h4{margin:0 0 4px;color:var(--amber);font-family:'Space Grotesk',sans-serif;}
@keyframes glow{0%,100%{box-shadow:0 0 0 rgba(255,180,84,0)}50%{box-shadow:0 0 28px rgba(255,180,84,.22)}}
@media (prefers-reduced-motion:reduce){.approve{animation:none;}}

[data-testid="stChatMessage"]{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:12px 16px;}
[data-testid="stChatInput"]{border-radius:16px;}
div.stButton>button{width:100%;border-radius:12px;border:1px solid var(--line);background:var(--surface);color:var(--text);transition:border-color .2s,transform .2s;}
div.stButton>button:hover{border-color:var(--sky);color:#fff;transform:translateY(-2px);}
div.stButton>button[kind="primary"],div.stButton>button[data-testid="stBaseButton-primary"]{background:var(--sky);border-color:var(--sky);color:#06101c;font-weight:600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# STATE
# =========================================================

ss = st.session_state
ss.setdefault("messages", [])   # LangChain messages (sent to the LLM)
ss.setdefault("log", [])        # what we draw on screen
ss.setdefault("pending", [])    # tool calls waiting for approval
ss.setdefault("auto", False)


def resolve(call, approved):
    name, args = call["name"], call["args"]
    if approved:
        try:
            out = str(TOOLS[name].invoke(args))
        except Exception as e:
            out = json.dumps({"error": str(e)})
        ss.log.append({"role": "tool", "name": name, "args": args, "content": out})
    else:
        out = "The user denied this tool call."
        ss.log.append({"role": "denied", "name": name, "args": args})
    ss.messages.append(ToolMessage(content=out, tool_call_id=call["id"]))


def run_agent():
    """Call the LLM until it either answers or needs a tool approval."""
    agent = get_agent()
    while True:
        try:
            result = agent.invoke(ss.messages)
        except Exception as e:
            ss.log.append({"role": "error", "content": str(e)})
            return
        ss.messages.append(result)

        if not result.tool_calls:
            ss.log.append({"role": "assistant", "content": result.content})
            return

        ss.pending = list(result.tool_calls)
        if not ss.auto:
            return
        while ss.pending:
            resolve(ss.pending.pop(0), True)


# =========================================================
# RENDER HELPERS
# =========================================================

ICONS = {"Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "🌦️", "Thunderstorm": "⛈️",
         "Snow": "❄️", "Mist": "🌫️", "Haze": "🌫️", "Fog": "🌫️", "Smoke": "🌫️", "Dust": "🌪️"}


def weather_card(d):
    place = ", ".join(x for x in (d["city"], d.get("country")) if x)
    tiles = [("Feels like", f"{d['feels_like']:.0f}°C"), ("Humidity", f"{d['humidity']}%"),
             ("Wind", f"{d['wind']} m/s")]
    tiles_html = "".join(f'<div class="tile"><span>{k}</span><b>{v}</b></div>' for k, v in tiles)
    return ('<div class="card"><div class="wx">'
            f'<div class="emo">{ICONS.get(d["condition"], "🌡️")}</div>'
            f'<div><div class="city">{esc(place)}</div><div class="desc">{esc(d["description"])}</div></div>'
            f'<div class="temp">{d["temp"]:.0f}°</div></div>'
            f'<div class="tiles">{tiles_html}</div></div>')


def news_card(d):
    links = "".join(
        f'<a class="news" href="{esc(n["url"])}" target="_blank">{esc(n["title"])}<br>'
        f'<small>{esc(urlparse(n["url"]).netloc.replace("www.", ""))}</small></a>'
        for n in d["news"])
    return f'<div class="card"><b>Latest in {esc(d["city"])}</b>{links}</div>'


def render_tool(entry):
    try:
        d = json.loads(entry["content"])
    except json.JSONDecodeError:
        d = None
    if isinstance(d, dict) and "error" in d:
        st.error(d["error"])
    elif isinstance(d, dict) and "temp" in d:
        st.markdown(weather_card(d), unsafe_allow_html=True)
    elif isinstance(d, dict) and "news" in d:
        st.markdown(news_card(d), unsafe_allow_html=True)
    else:
        st.code(entry["content"])


def render_log():
    for e in ss.log:
        role = e["role"]
        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(e["content"])
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🌆"):
                st.markdown(e["content"])
        elif role == "tool":
            with st.chat_message("assistant", avatar="🛠️"):
                st.caption(f"{e['name']}({json.dumps(e['args'])})")
                render_tool(e)
        elif role == "denied":
            with st.chat_message("assistant", avatar="🚫"):
                st.caption(f"You denied {e['name']}({json.dumps(e['args'])})")
        elif role == "error":
            st.error(f"Agent error: {e['content']}")


# =========================================================
# SIDEBAR
# =========================================================

KEYS = {"Groq": "GROQ_API_KEY", "OpenWeather": "OPENWEATHER_API_KEY", "Tavily": "TAVILY_API_KEY"}

with st.sidebar:
    st.markdown("### Control room")
    st.markdown("".join(
        f'<span class="chip {"on" if os.getenv(v) else "off"}">{k} {"ready" if os.getenv(v) else "key missing"}</span>'
        for k, v in KEYS.items()), unsafe_allow_html=True)
    st.caption(f"Model: {MODEL}")
    st.divider()
    st.toggle("Auto-approve tool calls", key="auto",
              help="Off = the agent asks you before every tool call.")
    used = sum(1 for e in ss.log if e["role"] == "tool")
    st.metric("Tools used this chat", used)
    if st.button("Clear chat"):
        ss.messages, ss.log, ss.pending = [], [], []
        st.rerun()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()


# =========================================================
# MAIN
# =========================================================

st.markdown('<div class="hero"><h1>City Intelligence</h1>'
            '<p>Live weather and headlines for any city. You approve every tool the agent uses.</p></div>',
            unsafe_allow_html=True)

if not ss.log:
    examples = ["Weather in Saharanpur", "News in Mumbai", "Weather and news in Delhi"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex):
            ss.queued = ex

render_log()

# Human-in-the-loop approval card
if ss.pending:
    call = ss.pending[0]
    st.markdown(
        '<div class="card approve"><h4>Agent wants to run a tool</h4>'
        f'<code>{esc(call["name"])}({esc(json.dumps(call["args"]))})</code></div>',
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    choice = None
    if c1.button("Approve", type="primary"):
        choice = True
    if c2.button("Deny"):
        choice = False
    if choice is not None:
        resolve(ss.pending.pop(0), choice)
        if not ss.pending:
            with st.spinner("Agent is thinking..."):
                run_agent()
        st.rerun()

prompt = st.chat_input("Ask about any city, e.g. weather in Jaipur", disabled=bool(ss.pending))
prompt = prompt or ss.pop("queued", None)

if prompt:
    ss.log.append({"role": "user", "content": prompt})
    ss.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.spinner("Agent is thinking..."):
        run_agent()
    st.rerun()