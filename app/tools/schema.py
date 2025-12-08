from pydantic import BaseModel
from typing import List, Optional

class ProductDetails(BaseModel):
    model: str
    name: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 0.0

class ProductSearchInput(BaseModel):
    model_number: Optional[str] = None
    query: Optional[str] = None
    search_type: Optional[str] = None

class ProductSearchOutput(BaseModel):
    products: List[ProductDetails]

class DocumentInfo(BaseModel):
    title: str
    url: Optional[str] = None
    source: Optional[str] = None

class DocumentSearchInput(BaseModel):
    query: str
    product_model: Optional[str] = None

class DocumentSearchOutput(BaseModel):
    documents: List[DocumentInfo]

class TicketInfo(BaseModel):
    id: str
    summary: Optional[str] = None
    similarity: float = 0.0

class PastTicketsSearchInput(BaseModel):
    query: str
    product_model: Optional[str] = None

class PastTicketsSearchOutput(BaseModel):
    tickets: List[TicketInfo]

class FinishToolInput(BaseModel):
    product_identified: bool
    product_details: Optional[ProductDetails] = None
    relevant_documents: List[DocumentInfo] = []
    relevant_images: List[str] = []
    past_tickets: List[TicketInfo] = []
    confidence: float = 0.0
    reasoning: Optional[str] = None
