from dotenv import load_dotenv
from typing import TypedDict
import os

from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate

# Load env
load_dotenv()

# Initialize LLM
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7
)

# -----------------------------
# STATE
# -----------------------------
class AgentState(TypedDict, total=False):
    query: str
    intent: str
    response: str


# -----------------------------
# NODE 1: CLASSIFY INTENT (LLM)
# -----------------------------
def classify_intent(state: AgentState):
    print("\n🔹 Classifying Intent")

    prompt = PromptTemplate.from_template("""
    Classify the user query into one of these categories:
    - tech
    - blog
    - general

    Query: {query}

    Only return one word: tech, blog, or general.
    """)

    chain = prompt | llm
    result = chain.invoke({"query": state.get("query", "")})

    intent = result.content.strip().lower()

    return {**state, "intent": intent}


# -----------------------------
# NODE 2A: TECH SUPPORT
# -----------------------------
def tech_support(state: AgentState):
    print("🔹 Tech Support Node")

    prompt = PromptTemplate.from_template("""
    Provide a technical solution for this query:
    {query}
    """)

    chain = prompt | llm
    result = chain.invoke({"query": state["query"]})

    return {**state, "response": result.content}


# -----------------------------
# NODE 2B: BLOG GENERATOR
# -----------------------------
def blog_generator(state: AgentState):
    print("🔹 Blog Generator Node")

    prompt = PromptTemplate.from_template("""
    Write a detailed blog about:
    {query}
    """)

    chain = prompt | llm
    result = chain.invoke({"query": state["query"]})

    return {**state, "response": result.content}


# -----------------------------
# NODE 2C: GENERAL CHAT
# -----------------------------
def general_chat(state: AgentState):
    print("🔹 General Chat Node")

    prompt = PromptTemplate.from_template("""
    Answer this casually and helpfully:
    {query}
    """)

    chain = prompt | llm
    result = chain.invoke({"query": state["query"]})

    return {**state, "response": result.content}


# -----------------------------
# ROUTER (CONDITIONAL LOGIC)
# -----------------------------
def route(state: AgentState):
    intent = state.get("intent", "general")

    if "tech" in intent:
        return "tech_node"
    elif "blog" in intent:
        return "blog_node"
    else:
        return "general_node"


# -----------------------------
# GRAPH
# -----------------------------
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("classify", classify_intent)
graph.add_node("tech_node", tech_support)
graph.add_node("blog_node", blog_generator)
graph.add_node("general_node", general_chat)

# Flow
graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route
)

graph.add_edge("tech_node", END)
graph.add_edge("blog_node", END)
graph.add_edge("general_node", END)

# Compile
app = graph.compile()


# -----------------------------
# RUN AGENT
# -----------------------------
if __name__ == "__main__":
    user_input = input("Enter your query: ")

    result = app.invoke({
        "query": user_input
    })

    print("\n🤖 FINAL RESPONSE:\n", result.get("response"))