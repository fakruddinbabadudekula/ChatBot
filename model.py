from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from os import getenv
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
  api_key=getenv("OPENROUTER_API_KEY"),
  base_url=getenv("OPENROUTER_BASE_URL"),
  model="openai/gpt-oss-20b:free",  # Replace with your desired model
  temperature=0.7,
 
)
