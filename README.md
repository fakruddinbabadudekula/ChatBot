# ChatBot


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
git clone https://github.com/fakruddinbabadudekula/ChatBot.git ChatBot
cd ChatBot
```

### 2️⃣ Create & activate a virtual environment

``` bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ If you want to use UV

``` bash
uv init

uv venv

.\.venv\Scripts\activate

uv add -r requirements.txt
```

> ✅ If you don't have a `requirements.txt`, here is a suggested one:
### requirements.txt
```bash 

faiss-cpu>=1.12.0
huggingface-hub>=0.36.0
langchain>=1.0.5
langchain-community>=0.4.1
langchain-core>=1.0.4
langchain-experimental>=0.4.0
langchain-google-genai>=3.0.3
langchain-huggingface>=1.0.1
langchain-openai>=1.0.2
langgraph>=1.0.3
langgraph-checkpoint-sqlite>=3.0.0
numpy>=2.3.4
pypdf2>=3.0.1
sentence-transformers>=5.1.2
streamlit>=1.51.0
torch>=2.9.1
transformers>=4.57.1

```

------------------------------------------------------------------------

## 🔑 Environment Setup

Create `.env` file in project root:

    GOOGLE_API_KEY=your_google_key
    OPENAI_API_KEY=your_openai_key(Open Router Api)
    OPENAI_MODEL=openai/gpt-oss-20b:free(you can also change see models in openrouter) 
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 

## 🔑 Geting OpenRouter API for free
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

The UI provides delete controls for: 
✅ Chat history which automatically deletes the associated resources and tools

Handled automatically through functions in `frontend.py`.

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   Support more file types
-   User authentication
-   Cloud persistent vector storage
-   Enhancement in retriever

## Demo Video
[▶️ Click to view demo video](assets\Demo.mp4)
------------------------------------------------------------------------

## 💡 Credits

Built using: - **Streamlit** - **LangGraph / LangChain** - **Google
Gemini / OpenAI** - **FAISS** - **HuggingFace Embeddings**

My speacial thanks to **Openrouter**
------------------------------------------------------------------------
