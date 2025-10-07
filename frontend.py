import streamlit as st
from main import chat
from langchain_core.messages import HumanMessage
import uuid
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

DATA_BASE="checkpoints.db"# Data Base file Name

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


# thread_id stores current thread_id if not create new one
if 'thread_id' not in st.session_state:
    thread_id=generate_thread_id()
    # create a obj form with thread_id and Name of the Chat for now and later it stored in history with new name
    st.session_state['thread_id']=thread_id

# chekpoint for sql to access the stored history
if 'checkpoint' not in st.session_state:
    conn=sqlite3.connect(DATA_BASE,check_same_thread=False)
    checkpoint=SqliteSaver(conn=conn)
    st.session_state['checkpoint']=checkpoint


# thread_history storing history of threads objects with id and name
if 'threads_history' not in st.session_state:
    st.session_state['threads_history']=[st.session_state['thread_id']]+retrieve_all_threads(checkpoint=st.session_state['checkpoint'])
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
    st.session_state['threads_history'].append(st.session_state['thread_id'])

for thread_id in st.session_state['threads_history']:
    if st.sidebar.button(thread_id):
        clean_chat_history()
        st.session_state['thread_id']=thread_id
        messages=load_conversation(thread_id=thread_id)
        for message in messages:
            st.session_state['messages'].append(
                {
                    'role':'user' if isinstance(message, HumanMessage) else 'assistant',
                    'content':message.content
                }
            )



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

    st.session_state['messages'].append({'role':'assistant','content':response})