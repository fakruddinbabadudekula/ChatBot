# ChatBot
Replica of ChatGPT but whit own code using langchain,langgraph and streamlit.

# 📌 AI Chatbot with File Upload (RAG + LangGraph + Streamlit)

This project is an interactive **Retrieval Augmented Generation (RAG)**
chatbot built using **Streamlit**, **LangGraph**, **FAISS**, and
**LLMs** (Google Gemini / OpenAI(Open Router Model)).\
Users can chat and upload documents (PDF/TXT). Those files are converted
into vector embeddings and stored thread-wise for context-aware answers.

------------------------------------------------------------------------

## ✅ Features

-   🧠 Chat with AI using thread-based session history
-   📄 Upload PDFs / TXT files as context
-   🔍 Automatic vector embedding creation for each thread
-   🗂️ Stored threads with ability to rename & delete
-   🧩 Tool-based retrieval using FAISS
-   💾 Chat history saved in SQLite via LangGraph Checkpointer

------------------------------------------------------------------------

## 📁 Project Structure

    project/
    │── frontend.py              # Streamlit UI
    │── workflow.py              # LangGraph workflow, retriever loading
    │── creating_vectore_store.py   # Creates FAISS vectorstore
    │── models.py                # LLM configuration (Google/OpenAI)
    │── checkpoints.db           # Auto-created DB for chat history
    │── thread_id_names.json     # Stores thread titles
    │── thread_id_uploads.json   # Stores document metadata
    │── vectorstores/            # Auto-created FAISS DB directories
    │── .env                     # API Keys

------------------------------------------------------------------------

## 🔧 Installation Guide

### 1️⃣ Clone the project

``` bash
git clone <your-repo-link>
cd project
```

### 2️⃣ Create & activate a virtual environment

``` bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

> ✅ If you don't have a `requirements.txt`, here is a suggested one:

# LangChain Core
langchain
langchain-core

# LangChain Community
langchain_community

# OpenAI Integration
langchain-openai
openai

# Anthropic Integration
langchain-anthropic

# Google Gemini (PaLM) Integration
langchain-google-genai
google-generativeai

# Hugging Face Integration
langchain-huggingface
transformers
huggingface-hub

# Environment Variable Management
python-dotenv

# Machine Learning Utilities
numpy
scikit-learn

# deep Learning
torch

# Youtube Transcript
youtube-transcript-api

# sentence-transformers for vector embedding model
sentence-transformers

# huggingface_hub[hf_xet] for fast downloading and storing models
huggingface_hub[hf_xet]

# faiss-cpu for vector storing
faiss-cpu

# streamlit 
streamlit

# langchain_experimental for experimental methods like shell tools 
langchain_experimental

# ddgs duckduckgo search for ddgs tool
ddgs

# langgraph
langgraph

# langgraph-checkpoint-sqllite for using sqllite
langgraph-checkpoint-sqlite

db-sqlite3

------------------------------------------------------------------------

## 🔑 Environment Setup

Create `.env` file in project root:

    GOOGLE_API_KEY=your_google_key
    OPENAI_API_KEY=your_openai_key(Open Router Api)
    OPENAI_MODEL=openai/gpt-oss-20b:free(you can also change)  

## 🔑 Getinng OpenRouter API for free
👉 https://youtu.be/E14hUveM4us?si=hBi2hFvrcrcrK9AX

*(Use whichever model you set in `models.py`)*

------------------------------------------------------------------------

## ▶️ Run the Application

``` bash
streamlit run frontend.py
```

Open the browser link provided (typically):\
👉 http://localhost:{port} port like 8000

------------------------------------------------------------------------

## 🗑️ Clearing Vectorstores & History

The UI provides delete controls for: ✅ Chat history\
✅ Vectorstore folder\
✅ Entries in JSON files & SQLite DB

Handled automatically through functions in `frontend.py`.

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   Support more file types
-   User authentication
-   Cloud persistent vector storage

------------------------------------------------------------------------

## 💡 Credits

Built using: - **Streamlit** - **LangGraph / LangChain** - **Google
Gemini / OpenAI** - **FAISS** - **HuggingFace Embeddings**

------------------------------------------------------------------------
