

import os
import json
import logging
import sqlite3
from dotenv import load_dotenv
# from typing import Literal
from functools import lru_cache
# ✅ LangGraph & LangChain
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ✅ LLMs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from models import llm,tool_llm
# ✅ Structured output
from pydantic import BaseModel, Field
from langchain_classic.tools.retriever import create_retriever_tool
load_dotenv()
logging.basicConfig(level=logging.INFO)

MAX_CHATS=20



# ==========================================================
# ✅ State Model for LangGraph
# ==========================================================

class State(MessagesState):
    thread_id: str = Field(description="Unique thread to track retrieval context")


# ==========================================================
# ✅ Embedding Loader
# ==========================================================

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ==========================================================
# ✅ Tool Loading
# ==========================================================
@lru_cache(maxsize=20)
def create_retriever_chached(thread_id: str):
    base_path = os.path.abspath("vectorstores")
    vector_path = os.path.abspath(os.path.join(base_path, os.path.basename(thread_id)))

    # Path must exist
    if not os.path.isdir(vector_path):
        raise FileNotFoundError(f"No vectorstore found for thread {thread_id}")

    # Ensure no traversal attacks
    if os.path.commonpath([vector_path, base_path]) != base_path:
        raise RuntimeError("Invalid vectorstore path")

    # Ensure FAISS index exists
    files = os.listdir(vector_path)
    if not any(fname.startswith("index") or fname.endswith(".faiss") for fname in files):
        raise FileNotFoundError("No FAISS index file found.")

    # Load FAISS once per thread and cache in RAM
    vectors = FAISS.load_local(
        vector_path,
        load_embeddings(),
        allow_dangerous_deserialization=True  # SAFE within your trust model
    )
    print("✅ Vector retriever created")
    return vectors.as_retriever(search_kwargs={"k": 8})


def retrieve_tool(thread_id, info):
    name = info.get("name")
    desc = info.get("description", "")
    # For caching the retriever to load the retriever into RAM: eleminates the process of loading the vectors again into RAM does for first time,load into L2 cache and cache it upto 20 bcz we put max 20
    retriever=create_retriever_chached(thread_id=thread_id)
    print("✅ retriever tool created")
    return create_retriever_tool(retriever, name, desc)


def load_tools(thread_id):
    try:
        with open("thread_id_uploads.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.error("❌ tools.json missing or invalid")
        return None

    if thread_id not in data:
        return None

    return [retrieve_tool(thread_id, info) for info in data[thread_id]]


# ==========================================================
# ✅ Prompts & Structured Output
# ==========================================================


GENERATE_PROMPT = (
  "Answer the following question please give the response to the finally asked question and take the history of messages and messages are {history}\n"
  "Here is the context {context} please give the response from this context if the context doesn't have the response then simply say i have no idea about the response or there is no exact data in the given context"
)



# ==========================================================
# ✅ Node Functions
# ==========================================================

# ✅ Call LLM (with optional tools)
def generate_query_or_respond(state: State) -> State:
    messages = state["messages"][-MAX_CHATS:]
    thread = state["thread_id"]

    if not messages:
        raise ValueError("⚠️ No messages found in state!")

    tools = load_tools(thread_id=thread)
    print("✅ Here are the tools which are passing to llm :",tools,type(tools))

    if tools:
        response = tool_llm.bind_tools(tools=tools).invoke(messages)
        if response.content:
            response.additional_kwargs['final_node']=True
            print(f"✅ genereate_query_or_respond resoponse with tool llm  => {response}") 

    else:
        response = llm.invoke(messages)
        response.additional_kwargs['final_node']=True
        print(f"✅ genereate_query_or_respond resoponse llm=> {response}") 

    return {"messages": [response]}




# ✅ Execute tool calls
def execute_tool(state: State):
    last = state["messages"][-1]

    if not getattr(last, "tool_calls", None):
        raise ValueError("⚠️ No tool call found in last message!")

    call = last.tool_calls[0]
    tool_name = call["name"]
    # Check the arguments or validate the args but no need for this bcz it only have query args right
    args = call["args"]
    thread_id = state.get("thread_id")
    print(f"✅ llm calls the tool with the name and args=>{tool_name} and {args} ")
    # Checking if the tool is in listed tool for the given thread_id
    available_tools = load_tools(thread_id) or []
    available_names = {t.name for t in available_tools}
    print(f"✅ Available tool names for this thread_id => {available_names}")
    if tool_name not in available_names:
        logging.warning("Blocked tool call to unknown tool '%s' for thread '%s'", tool_name, thread_id)
        raise ValueError("Attempted to call an unknown or unauthorized tool.")


    result_tool = retrieve_tool(thread_id, {"name": tool_name})
    output = result_tool.run(args)
    print(f"✅ response from tool {tool_name} => {output}")


    
    return {
        "messages":[
            ToolMessage(content=output, name=tool_name, tool_call_id=call.get("id"))
        ]
    }


# ✅ Final RAG Answer
def generate_answer(state: State):
    history = state["messages"][-MAX_CHATS:-1]
    last_message = state["messages"][-1]# again converting into json.

    context=last_message.content
    answer = llm.invoke([
        HumanMessage(content=GENERATE_PROMPT.format(history=history, context=context))
    ])
    print(f"✅ final generated answer => {answer}")
    answer.additional_kwargs['final_node']=True


    return {"messages": [answer]}


# ==========================================================
# ✅ LLM Config
# ==========================================================




# ==========================================================
# ✅ Checkpointing
# ==========================================================

db = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpoint = SqliteSaver(db)


# ==========================================================
# ✅ StateGraph Wiring
# ==========================================================

def tools_condition(state: State):
    last = state["messages"][-1]
    return "tool_node" if isinstance(last, AIMessage) and last.tool_calls else END


workflow = StateGraph(State)

workflow.add_node("agent", generate_query_or_respond)

workflow.add_node("tool_node", execute_tool)
workflow.add_node("answer", generate_answer)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tool_node": "tool_node", END: END}
)
# workflow.add_edge('tool_node','grade')

workflow.add_edge(
    "tool_node",
   'answer'
)

workflow.add_edge("answer", END)

graph = workflow.compile(checkpointer=checkpoint)