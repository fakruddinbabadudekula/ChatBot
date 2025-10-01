import streamlit as st
from main import chat
from langchain_core.messages import HumanMessage

if 'messages' not in st.session_state:
    st.session_state['messages']=[]

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

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
            # config={'configurable': {'thread_id': 'thread-1'}},
            
            # Stream mode: stream full messages instead of raw tokens
            stream_mode='messages'
        )
    )

    st.session_state['messages'].append({'role':'assistant','content':response})