import operator
from typing import Annotated, TypedDict, Union
from langchain_ollama import ChatOllama
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END

# 1. Define the State
class AgentState(TypedDict):
    company: str
    research_data: str
    final_report: str

# 2. Define the Nodes
search_tool = DuckDuckGoSearchRun()
model = ChatOllama(model="qwen3:4b")

def research_node(state: AgentState):
    print(f"--- Researching: {state['company']} ---")
    results = search_tool.run(f"Latest business news for {state['company']}")
    return {"research_data": results}

def summarize_node(state: AgentState):
    print("--- Generating Report ---")
    prompt = f"Based on this news: {state['research_data']}, write a 3-bullet MarTech strategy for {state['company']}."
    response = model.invoke(prompt)
    return {"final_report": response.content}

# 3. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_node("writer", summarize_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

agent_app = workflow.compile()


import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

# 1. Define GraphQL Types
@strawberry.type
class IntelligenceReport:
    company: str
    report: str

@strawberry.type
class Query:
    @strawberry.field
    async def get_insight(self, company: str) -> IntelligenceReport:
        # Run the LangGraph Agent
        inputs = {"company": company}
        result = await agent_app.ainvoke(inputs)
        return IntelligenceReport(
            company=company, 
            report=result["final_report"]
        )

# 2. Initialize FastAPI
schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")

# Add a simple REST health check (Standard Enterprise practice)
@app.get("/health")
def health():
    return {"status": "agent_online"}