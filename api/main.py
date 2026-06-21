from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lexrag.config import settings
from lexrag.logger import get_logger
from .routes import router

logger = get_logger(__name__)

app = FastAPI(
    title="LexRAG Enterprise API",
    description="High-performance Graph RAG backend powered by LangGraph, Neo4j, and Qdrant.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include application routes
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting LexRAG API", version="1.0.0")
    
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
