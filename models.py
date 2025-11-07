from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENAI_MODEL=os.getenv("OPENAI_MODEL")

if OPENAI_API_KEY in ["","OPENAI_API_KEY"] or GOOGLE_API_KEY in ["","GOOGLE_API_KEY"]:
    raise RuntimeError("⚠️ Missing OPENAI_API_KEY or GOOGLE_API_KEY in .env file!")

llm = ChatOpenAI(
  api_key=OPENAI_API_KEY,
  base_url=OPENROUTER_URL,
  model=OPENAI_MODEL,  # Replace with your desired model
  temperature=0.7,
 
)
tool_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)
