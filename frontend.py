import streamlit as st
# from main import chat,checkpoint
from workflow import graph as chat,checkpoint
from langchain_core.messages import HumanMessage,AIMessage
import uuid
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from creating_vectore_store import create_vector_store
from models import llm
import json
import os
import re
import logging
from dotenv import load_dotenv
import shutil  
logging.basicConfig(level=logging.INFO)
DATA_BASE="checkpoints.db"# Data Base file Name
JSON_FILE_NAME="thread_id_names.json"
JSON_UPLOAD="thread_id_uploads.json"
load_dotenv()

def creating_json_files():
    thread_id_uploads = "thread_id_uploads.json"
    if not os.path.exists(thread_id_uploads):
        with open(thread_id_uploads, "w") as f:
            json.dump({}, f)
        print(f"✅ Created file: {thread_id_uploads}")
    thread_id_names="thread_id_names.json"
    if not os.path.exists(thread_id_names):
        with open(thread_id_names, "w") as f:
            json.dump({"threads_with_names":[]}, f)
        print(f"✅ Created file: {thread_id_names}")

creating_json_files()

def clean_filename(filename: str) -> str:
    base = os.path.splitext(filename)[0]  # Remove extension
    cleaned = re.sub(r'[^A-Za-z0-9]+', '_', base)  # Replace non-alphanumeric with _
    return cleaned.strip('_')
# helper func to load the resources file
def load_resources(json_path=JSON_UPLOAD):
    """Load data from the JSON file."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # Return empty dict if file doesn't exist
    
def save_data(data):
    """Save data back to the JSON file."""
    with open(JSON_UPLOAD, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ saved data successfully in {JSON_UPLOAD}")

# helper func to add the resources with thread_id to the JSON_UPLOAD file
def add_resources(resources,thread_id):
    data=load_resources()
    if thread_id not in data:
        data[thread_id]=[]
    data[thread_id].append(resources)
    save_data(data=data)

# helper to generate new thread_id
def generate_thread_id():
    return str(uuid.uuid4())

# return list of threads which are currently stored
def retrieve_all_threads(checkpoint):
    all_threads = set()
    for checkpoint in checkpoint.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)

# clearing the chat sessions
def clean_chat_history():
    st.session_state['messages']=[]
# Loading the Chat history from the specified thred_id
def load_conversation(thread_id):
    state = chat.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    messages=state.values.get('messages', [])
    filterd_msges=[]
    for msg in messages:
        if msg.type=="human" or msg.additional_kwargs.get('final_node')==True:
            if isinstance(msg.content,list):
                content=""
                for text in msg.content:
                    if isinstance(text,str):
                        content+=text
                    elif (text.get("type",None)=="text" and text.get('text',None)):
                        content+=text.get('text')
                msg.content=content


            filterd_msges.append(msg)
        
    return filterd_msges

def create_name(messages):
  
    prompt="""You are a smart assistant that generates short, clear titles for chat conversations.

I will give you the first user message and the first AI response of a conversation.

Based on them, create a short (max 6 words) descriptive title that summarizes the main topic of the conversation.
Avoid generic words like "Chat" or "Conversation".
Capitalize main words like a title (e.g., "Machine Learning Basics", "Travel Plans to Japan").

User message:
"{query}"

AI response:
"{response}"

Return only the title, nothing else.
"""
    
    final_input=prompt.format(query=messages['query'],response=messages['response'])
    name=llm.invoke(final_input).content
    return str(name)
# removde create_description instead uses the file type and file name to create description.

# def create_description(text):
#     prompt = """
# You are an assistant that writes concise lesson descriptions.

# I will give you the first portion of text extracted from a lesson document.

# Your task:
# - Write a clear and informative description of what the lesson is about.
# - Summarize the key purpose or learning objective.
# - Write in single sentence(max ~80words).
# - Use simple and professional language suitable for a course description.
# - Do not mention the document or extraction process.

# Lesson text:
# "{text}"

# Return only the description.
# """

#     final_input = prompt.format(text=text
#                                 )
#     result = llm.invoke(final_input).content
#     return str(result)


def get_names(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)

    return data['threads_with_names']


def update_names(data,file_name=JSON_FILE_NAME):
    final_data={
                "threads_with_names":data
                }
    with open(file_name, "w") as f:
        json.dump(final_data, f, indent=4)

def update_thread_name(threads, thread_id, new_name):
    for thread in threads:
        if thread["thread_id"] == thread_id:
            thread["name"] = new_name
            break  # stop after updating
    return threads

def load_uploaded_files(thread_id):
    try:
        with open("thread_id_uploads.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.error("❌ thread_id_uploads.json missing or invalid")
        return []
    if thread_id not in data:
        return []
    names=[info.get('name') for info in data[thread_id]]
    return names
# thread_id stores current thread_id if not create new one
if 'thread_id' not in st.session_state:
    thread_id=generate_thread_id()
    # create a obj form with thread_id and Name of the Chat for now and later it stored in history with new name
    st.session_state['thread_id']=thread_id

# thread_history storing history of threads objects with id and name
if 'threads_history' not in st.session_state:
    thread_id_names=get_names(file_name=JSON_FILE_NAME)
    current_thred_id_name={'thread_id':st.session_state['thread_id'],'name':st.session_state['thread_id']}


    st.session_state['threads_history']=[current_thred_id_name]+thread_id_names
    # For sql retrieve all threads 
    # graph automatically fetch the history of messages from the sql when we pass the thread id it like inmemory checkpointer
    # st.session_state['threads_history']=st.session_state['threads_history'].append(retrieve_all_threads(checkpoint=st.session_state['checkpoint']))

# history of messages for a specified thread_id
if 'messages' not in st.session_state:
    st.session_state['messages']=[]


if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

st.set_page_config(page_title="Chatbot with File Upload", layout="wide")

if st.sidebar.button("➕", key="sidebar_upload_button"):
    st.session_state.show_uploader = not st.session_state.show_uploader

if "stored_files" not in st.session_state:
    st.session_state.stored_files = load_uploaded_files(thread_id=st.session_state['thread_id'])
st.sidebar.title("ChatBot")

if st.session_state.show_uploader:
    uploaded_files = st.sidebar.file_uploader(
        "Upload resources",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="sidebar_file_uploader",
    )

    if uploaded_files:
        with st.spinner("🔄 Saving your document & generating embeddings..."):
            for f in uploaded_files:
                # Read the file content depending on its type
                if f.type == "application/pdf":
                    from PyPDF2 import PdfReader
                    reader = PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                elif f.type == "text/plain":
                    text = f.read().decode("utf-8", errors="ignore")
                else:
                    st.warning(f"Unsupported file type: {f.type}")
                    continue
                
                # Store basic file info
                name=clean_filename(f.name)
                file_description=f"A tool that fetch the related information about the user requested data and the document type is {f.type} and the name of the document is {name}"
                file_data = {
                    "name": name,
                    "type": f.type,
                    "description":file_description
                    
                }
                

                st.session_state.stored_files.append(file_data.get('name'))
                add_resources(file_data, st.session_state["thread_id"])

                # ✅ Call your vector creation function
                tool = create_vector_store(
                    docs=text,
                    thread_id=st.session_state["thread_id"],
                )

                st.success(f"Vector store created for {f.name}")

            st.session_state.show_uploader = False
            st.rerun()



if st.session_state.stored_files:
    st.sidebar.markdown("### Uploaded files (simulated)")
    for file_names in st.session_state.stored_files:
        st.sidebar.markdown(f"- {file_names}")   

# Creating the New Chat
if st.sidebar.button("New Chat"):
    clean_chat_history()#clearning history of messages
    st.session_state['stored_files']=[]
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    st.session_state['threads_history'].append({'thread_id':st.session_state['thread_id'],'name':st.session_state['thread_id']})
    st.rerun()

def delete_vectors_from_json_storage(thread_id,json_path=JSON_UPLOAD,vector_dir="vectorstores"):
    st.session_state['stored_files']=[]
    data=load_resources(json_path=JSON_UPLOAD)
    
    if thread_id in data:
            # ✅ Secure path construction
        safe_id = os.path.basename(thread_id)  # Remove traversal elements
        base_path = os.path.abspath(vector_dir)
        vector_path = os.path.abspath(os.path.join(vector_dir, safe_id))

        # ✅ Ensure directory exists before deletion
        if not os.path.isdir(vector_path):
            logging.warning(f"⚠️ Vector directory not found: {vector_path}")
            return

        # ✅ Ensure vector_path is inside base_path
        if os.path.commonpath([vector_path, base_path]) != base_path:
            logging.error(f"❌ Security Alert: Path traversal attempt blocked — {vector_path}")
            return

        # ✅ Safe delete
        shutil.rmtree(vector_path)
        logging.info(f"✅ Vectorstore deleted: {vector_path}")
        del data[thread_id]
        save_data(data=data) 
        

def delete_thread_from_sql(thread_id):
   checkpoint.delete_thread(thread_id)

# --- Sidebar threads list ---
def delete_thread_from_json(thread_id, json_path=JSON_FILE_NAME):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Filter out the deleted thread
        data["threads_with_names"] = [
            t for t in data.get("threads_with_names", [])
            if t.get("thread_id") != thread_id
        ]

        # Save back to file
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    except FileNotFoundError:
        print(f"⚠️ JSON file not found: {json_path}")

# --- Sidebar threads ---
for thread in st.session_state['threads_history']:
    if thread['thread_id'] == thread['name']:
        continue

    col1, col2 = st.sidebar.columns([4, 1])

    with col1:
        # Load chat button
        if st.button(thread['name'], key=f"load_{thread['thread_id']}"):
            clean_chat_history()
            st.session_state['thread_id'] = thread['thread_id']
            st.session_state['stored_files']=load_uploaded_files(thread_id=st.session_state['thread_id'])
            messages = load_conversation(thread_id=thread['thread_id'])
            for message in messages:
                st.session_state['messages'].append(
                    {
                        'role': 'user' if isinstance(message, HumanMessage) else 'assistant',
                        'content': message.content
                    }
                )
            if st.session_state.stored_files:
                st.sidebar.markdown("### Uploaded files (simulated)")
                for file_names in st.session_state.stored_files:
                    st.sidebar.markdown(f"- {file_names}")   

    with col2:
        # Delete button
        if st.button("🗑️", key=f"delete_{thread['thread_id']}"):
            import streamlit.components.v1 as components


            # Delete from SQL and JSON
            clean_chat_history()
            delete_thread_from_sql(thread['thread_id'])
            delete_thread_from_json(thread['thread_id'])
            delete_vectors_from_json_storage(thread_id=thread['thread_id'])

            st.session_state['threads_history'] = [
                t for t in st.session_state['threads_history']
                if t['thread_id'] != thread['thread_id']
            ]
            alert_js = """alert("Successfully deleted the Chat");"""
            components.html(f"<script>{alert_js}</script>", height=0)
            st.rerun()

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state['messages'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)


    with st.chat_message("assistant"):
        
        def stream_response():
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "thread_id": st.session_state['thread_id']
            }

            for message_chunk, metadata in chat.stream(
                inputs,
                config=CONFIG,
                stream_mode="messages"
            ):
                if message_chunk.content and isinstance(message_chunk, AIMessage) and metadata.get("langgraph_node") in ['answer','agent']:
                    if isinstance(message_chunk.content,list):
                        for text in message_chunk.content:
                            content=""
                            if isinstance(text,str):
                                yield text
                            elif (text.get("type",None)=="text" and text.get('text',None)):
                                yield text.get('text')
                    else:
                        yield message_chunk.content
        try:
            with st.spinner("🤖 Thinking..."):
                response = st.write_stream(stream_response)
        except Exception as e:
            print(e)
            st.error(f"⚠️ Error occurred: {e}")
            response=e



    if len(st.session_state['messages'])==1:
        thread_name=create_name({
            'query':user_input,
            'response':response
        })
        thread_id_names=get_names(file_name=JSON_FILE_NAME)
        thread_id_names.append({
            'thread_id':st.session_state['thread_id'],
            'name':thread_name
        })
        update_names(data=thread_id_names)
        st.session_state['threads_history']=update_thread_name(st.session_state['threads_history'],st.session_state['thread_id'],new_name=thread_name)

    st.session_state['messages'].append({'role':'assistant','content':response})