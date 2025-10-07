from langgraph.graph import StateGraph,START,END,MessagesState
from model import llm
from langchain_core.messages import HumanMessage,AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Create SQLite connection
# sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread. The object was created in thread id 8948 and this is thread id 3776.
# so pass the check_same_thread=False
sql_connector = sqlite3.connect("checkpoints.db",check_same_thread=False)

def llm_node(state:MessagesState):
    query=state['messages']
    response=llm.invoke(query)
    return {
        'messages':response
    }


# checkpoint=InMemorySaver()

checkpoint=SqliteSaver(conn=sql_connector)

graph=StateGraph(MessagesState)
graph.add_node("llm_node",llm_node)
graph.add_edge(START,'llm_node')
graph.add_edge('llm_node',END)
chat=graph.compile(checkpointer=checkpoint)

