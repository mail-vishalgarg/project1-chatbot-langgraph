from state import State
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage


#Define the chat model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define the fucntion of the Node
def chatbot(state:State)->State:
    user_question = state["question"]
    response = llm.invoke(user_question)
    return {
        **state,
        "answer":response.content
    }

# Create the graph

graph_builder = StateGraph(State)
# add node
graph_builder.add_node("LLM Response", chatbot)

#Add Edges/Entry points
graph_builder.add_edge(START, "LLM Response")

#Add final edge/Exit point
graph_builder.add_edge("LLM Response", END)

# Compile the graph
graph = graph_builder.compile()

# print the graph
print(graph.get_graph().draw_mermaid())

#Below code will generate a PNG file of the graph
#graph.get_graph().draw_mermaid_png(output_file_path="final_graph.png")

#Execute the graph
if __name__ == "__main__":
    user_question = "who will win fifa in 2026?"
    final_state = graph.invoke({"question": user_question})
    print(final_state)