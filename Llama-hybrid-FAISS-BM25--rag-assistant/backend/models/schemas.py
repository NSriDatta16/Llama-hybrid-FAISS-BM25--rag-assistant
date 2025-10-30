from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: Optional[float] = None
    model: Optional[str] = None      # <-- add this

class RetrievedChunk(BaseModel):
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = {}

class AskResponse(BaseModel):
    answer: str
    citations: List[RetrievedChunk]
    used_alpha: float
