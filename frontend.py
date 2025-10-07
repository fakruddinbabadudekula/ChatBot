import streamlit as st
from main import chat
from langchain_core.messages import HumanMessage
import uuid

# helper to generate new thread_id
def generate_thread_id():
    return str(uuid.uuid4())




def clean_chat_history():
    st.session_state['messages']=[]

def load_conversation(thread_id):
    state = chat.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# thread_id stores current thread_id if not create new one
if 'thread_id' not in st.session_state:
    thread_id=generate_thread_id()
    # create a obj form with thread_id and Name of the Chat for now and later it stored in history with new name
    st.session_state['thread_id']={'thread_id':thread_id,'name':thread_id[:10]}

# thread_history storing history of threads objects with id and name
if 'threads_history' not in st.session_state:
    st.session_state['threads_history']=[st.session_state['thread_id']]




if 'messages' not in st.session_state:
    st.session_state['messages']=[]




st.sidebar.title("ChatBot")

if st.sidebar.button("New Chat"):
    clean_chat_history()
    thread_id=generate_thread_id()
    st.session_state['thread_id']={'thread_id':thread_id,'name':thread_id[:10]}
    st.session_state['threads_history'].append(st.session_state['thread_id'])

for thread in st.session_state['threads_history']:
    if st.sidebar.button(thread['name']):
        clean_chat_history()
        st.session_state['thread_id']={'thread_id':thread['thread_id'],'name':thread['name']}
        messages=load_conversation(thread_id=thread['thread_id'])
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


CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']['thread_id']}}
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