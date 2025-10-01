from langgraph.graph import StateGraph,START,END,MessagesState
from model import llm
from langchain_core.messages import HumanMessage

def llm_node(state:MessagesState):
    query=state['messages']
    response=llm.invoke(query)
    return {
        'messages':response
    }

graph=StateGraph(MessagesState)
graph.add_node("llm_node",llm_node)
graph.add_edge(START,'llm_node')
graph.add_edge('llm_node',END)
chat=graph.compile()