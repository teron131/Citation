import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GraphState(BaseModel):
    sentence: str = Field(description="A sentence to be considered.")
    # full_content: str = Field(description="The full content of the paper.")


def get_more_context(state: GraphState) -> GraphState:
    return state


def write_query(state: GraphState) -> GraphState:
    return state


def retrieve_arxiv(state: GraphState) -> GraphState:
    return state


def verify_similarity(state: GraphState) -> GraphState:
    return state


def get_full_papers(state: GraphState) -> GraphState:
    return state


def select_citation(state: GraphState) -> GraphState:
    return state


def write_citation(state: GraphState) -> GraphState:
    return state


builder = StateGraph(GraphState)
builder.add_node(select_snippet)
builder.add_node(get_more_context)
builder.add_node(write_query)
builder.add_node(retrieve_arxiv)
builder.add_node(verify_similarity)
builder.add_node(get_full_papers)
builder.add_node(select_citation)
builder.add_node(write_citation)

builder.set_entry_point("select_snippet")
builder.add_edge("select_snippet", "write_query")
builder.add_edge("select_snippet", "get_more_context")
builder.add_edge("get_more_context", "write_query")
builder.add_edge("write_query", "retrieve_arxiv")
builder.add_edge("retrieve_arxiv", "verify_similarity")
builder.add_conditional_edges(
    "verify_similarity",
    verify_similarity,
    {
        True: "get_full_papers",
        False: "write_query",
    },
)
builder.add_edge("get_full_papers", "select_citation")
builder.add_edge("select_citation", "write_citation")
builder.set_finish_point("write_citation")

graph = builder.compile()


# Save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
