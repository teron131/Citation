import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GraphState(TypedDict):
    sentence: str  # A sentence to be considered
    full_content: str  # The full content of the paper
    similarity: float  # Cosine similarity of embeddings


def get_more_context(state: GraphState) -> GraphState:
    return state


def write_query(state: GraphState) -> GraphState:
    return state


def retrieve_arxiv(state: GraphState) -> GraphState:
    return state


def calculate_similarity(text1: str, text2: str) -> float:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    doc_embedding = embeddings.embed_query(text1)
    query_embedding = embeddings.embed_query(text2)
    similarity = cosine_similarity([doc_embedding], [query_embedding]).item()
    return similarity


def verify_similarity(state: GraphState):
    similarity = calculate_similarity(state["sentence"], state["full_content"])
    state["similarity"] = similarity
    return "yes" if state["similarity"] > 0.01 else "no"


def get_full_papers(state: GraphState) -> GraphState:
    return state


def select_citation(state: GraphState) -> GraphState:
    return state


def write_citation(state: GraphState) -> GraphState:
    return state


builder = StateGraph(GraphState)
builder.add_node("get_more_context", get_more_context)
builder.add_node("write_query", write_query)
builder.add_node("retrieve_arxiv", retrieve_arxiv)
builder.add_node("get_full_papers", get_full_papers)
builder.add_node("select_citation", select_citation)
builder.add_node("write_citation", write_citation)

builder.add_edge(START, "get_more_context")
builder.add_edge("get_more_context", "write_query")
builder.add_edge("write_query", "retrieve_arxiv")
builder.add_conditional_edges(
    "retrieve_arxiv",
    verify_similarity,
    {
        "yes": "get_full_papers",
        "no": "write_query",
    },
)
builder.add_edge("get_full_papers", "select_citation")
builder.add_edge("select_citation", "write_citation")
builder.add_edge("write_citation", END)
graph = builder.compile()

# Save the graph visualization
graph_image = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)

if __name__ == "__main__":
    full_content = open("2305.05665v2.txt").read()

    result = graph.invoke(
        {
            "sentence": "In multilingual neural machine translation, a similar phenomenon to the emergence behavior of IMAGEBIND is commonly observed and utilized: if languages are trained in the same latent space through learned implicit bridging, translation can be done between language pairs on which no paired data is provided",
            "full_content": full_content,
        }
    )
    print(result)
