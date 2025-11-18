
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os


def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_vector_store(docs, thread_id,
                        chunk_size=2000, chunk_overlap=150):
    base_path = os.path.abspath("vectorstores")
    index_path = os.path.abspath(os.path.join(base_path, os.path.basename(thread_id)))
    print(f"✅ index_path for vector=> {index_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=["\n\n", "\n", ".", " ", ""])

    # Split docs into chunks
    chunks = splitter.create_documents([docs])
    print(f"✅ Created {len(chunks)} chunks")

    embeddings = load_embeddings()

    # Case 1: Vectorstore exists — LOAD & APPEND
    if os.path.exists(index_path):
        print("🔄 Existing vectorstore found — adding new chunks...")

        vectors = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectors.add_documents(chunks)
        vectors.save_local(index_path)

    # Case 2: Vectorstore does NOT exist — CREATE NEW
    else:
        print("🆕 Creating new vectorstore...")
        os.makedirs(base_path, exist_ok=True)
        vectors = FAISS.from_documents(chunks, embeddings)
        vectors.save_local(index_path)

    print("✅ Vector store updated successfully!")
    return vectors
