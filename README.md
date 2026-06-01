# LangGraph docs chatbot

Small hello-world agent: LangGraph + Gemini, answers from [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your key in `.env` as `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey).

## Run

```bash
python chatbot.py
```

Try the questions in `LANGGRAPH_QUESTIONS.md`.

## Files

```
chatbot.py
LANGGRAPH_QUESTIONS.md
requirements.txt
.env.example
```

All code is in `chatbot.py`.
# Hexaware-Langgraph-Test
