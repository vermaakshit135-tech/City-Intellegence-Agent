# Runnable Testing

This project is a small collection of LangChain and LLM experiments, with a Streamlit-based city intelligence assistant as the main demo.

## Project overview

The repository includes:

- a browser-based agent in `webpage.py` that can answer questions about city weather and latest news
- a CLI version in `Agents.py` that performs the same tool-calling flow in the terminal
- several smaller experiments for Mistral, Tavily search, custom tools, and sequential chains

## Main app

The main application is the Streamlit interface in `webpage.py`.

It uses:

- Groq as the LLM backend
- OpenWeather API to get live weather
- Tavily API to fetch recent city news
- a human-in-the-loop approval flow before running tools

Example prompts:

- Weather in Saharanpur
- News in Mumbai
- Weather and news in Delhi

## Project files

- `webpage.py` — Streamlit city assistant with tool approval
- `Agents.py` — terminal-based city intelligence agent
- `mistral.py` — quick Mistral API test
- `newssummerize.py` — AI news summarization using Tavily search + Mistral
- `owntool.py` — example custom tool implementation
- `toolcalling.py` — simple tool-calling example with Mistral
- `sequential.py` — example of a sequential / parallel chain workflow
- `requirements.txt` — Python dependencies

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project root with the required API keys:

```env
GROQ_API_KEY=your_groq_key
OPENWEATHER_API_KEY=your_openweather_key
TAVILY_API_KEY=your_tavily_key
MISTRAL_API_KEY=your_mistral_key
```

Notes:

- `webpage.py` checks for `GROQ_API_KEY`, `OPENWEATHER_API_KEY`, and `TAVILY_API_KEY`
- Mistral-related scripts use `MISTRAL_API_KEY`
- If a key is missing, the relevant script or app may fail or show an error

## Run the app

Start the Streamlit app:

```bash
streamlit run webpage.py
```

Then open the local URL shown in the terminal in your browser.

## Run other examples

```bash
python Agents.py
python mistral.py
python newssummerize.py
python owntool.py
python toolcalling.py
python sequential.py
```

## Notes

This repository is mostly a learning / testing environment for experimenting with:

- LangChain tools
- LLM tool calling
- API integrations
- prompt chaining
- Streamlit chat interfaces

## License

This project is for educational and personal experimentation purposes.
