from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agent.forensic_agent import ForensicRAGAgent

router = APIRouter(prefix="/agent", tags=["Forensic RAG Agent"])

class AgentQueryRequest(BaseModel):
    query: str

class AgentQueryResponse(BaseModel):
    success: bool
    query: str
    answer: str

@router.post("/query", response_model=AgentQueryResponse)
def query_agent(payload: AgentQueryRequest):
    """
    Submits a natural language query to the Forensic RAG Agent.
    Retrieves project status, dataset health records, and model training metrics to synthesize a response.
    """
    try:
        agent = ForensicRAGAgent()
        answer = agent.answer_query(payload.query)
        return AgentQueryResponse(
            success=True,
            query=payload.query,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forensic agent failed to compile response: {str(e)}"
        )
