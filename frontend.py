import streamlit as st
from main import chat,checkpoint
from langchain_core.messages import HumanMessage
import uuid
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from model import llm
import json

DATA_BASE="checkpoints.db"# Data Base file Name
JSON_FILE_NAME="thread_id_names.json"

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
    return state.values.get('messages', [])

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




st.sidebar.title("ChatBot")

# Creating the New Chat
if st.sidebar.button("New Chat"):
    clean_chat_history()#clearning history of messages
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    st.session_state['threads_history'].append({'thread_id':st.session_state['thread_id'],'name':st.session_state['thread_id']})


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
            messages = load_conversation(thread_id=thread['thread_id'])
            for message in messages:
                st.session_state['messages'].append(
                    {
                        'role': 'user' if isinstance(message, HumanMessage) else 'assistant',
                        'content': message.content
                    }
                )

    with col2:
        # Delete button
        if st.button("🗑️", key=f"delete_{thread['thread_id']}"):
            # Delete from SQL and JSON
            delete_thread_from_sql(thread['thread_id'])
            delete_thread_from_json(thread['thread_id'])

            st.session_state['threads_history'] = [
                t for t in st.session_state['threads_history']
                if t['thread_id'] != thread['thread_id']
            ]

            st.rerun()

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
user_input = st.chat_input()


if user_input:
    st.session_state['messages'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)


    with st.chat_message('assistant'):

        response = st.write_stream(
        # Generator comprehension: extract only the content (text) 
        # from each streamed message chunk
        message_chunk.content 
        for message_chunk, metadata in chat.stream(
            
            # Input to the chatbot: one user message
            {'messages': [HumanMessage(content=user_input)]},
            
            # Config: thread_id keeps conversation state across turns
            config=CONFIG,
            
            # Stream mode: stream full messages instead of raw tokens
            stream_mode='messages'
        )
    )
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