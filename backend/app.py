"""
FastAPI Backend for RAG System
Provides REST API endpoints for document upload and question answering
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from data_loader import DataLoader, TextChunker
from embedding import EmbeddingGenerator
from vector_store import FAISSVectorStore
from retrieval import RAGSystem

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation using Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
VECTORSTORE_PATH = os.getenv('VECTORSTORE_PATH', './vectorstore')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 100))

vector_store = FAISSVectorStore(store_path=VECTORSTORE_PATH)
embedding_generator = EmbeddingGenerator()
rag_system = RAGSystem(vector_store, embedding_generator)

# Request/Response models
class UploadResponse(BaseModel):
    message: str
    chunks_count: int
    embeddings_count: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list


# Health check endpoint
@app.get("/health")
def health_check():
    """Check API health"""
    return {
        "status": "healthy",
        "vectorstore_size": vector_store.get_size()
    }


# Upload and process document endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF or TXT)
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"./temp/{file.filename}"
        os.makedirs("./temp", exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # Load and clean document
        text, source_type = DataLoader.load_document(temp_path)
        
        # Chunk text
        chunks = TextChunker.chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        
        if not chunks:
            raise ValueError("No content extracted from document")
        
        # Generate embeddings
        embeddings = embedding_generator.batch_generate_embeddings(chunks)
        
        # Add to vector store
        metadata = [{"source": file.filename, "type": source_type} for _ in chunks]
        vector_store.add_embeddings(embeddings, chunks, metadata)
        
        # Cleanup
        os.remove(temp_path)
        
        return UploadResponse(
            message=f"Successfully processed {file.filename}",
            chunks_count=len(chunks),
            embeddings_count=len(embeddings)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Upload from URL endpoint
@app.post("/upload-url", response_model=UploadResponse)
async def upload_from_url(url: str):
    """
    Load and process document from URL
    """
    try:
        # Load document from URL
        text, source_type = DataLoader.load_document(url)
        
        # Chunk text
        chunks = TextChunker.chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        
        if not chunks:
            raise ValueError("No content extracted from URL")
        
        # Generate embeddings
        embeddings = embedding_generator.batch_generate_embeddings(chunks)
        
        # Add to vector store
        metadata = [{"source": url, "type": source_type} for _ in chunks]
        vector_store.add_embeddings(embeddings, chunks, metadata)
        
        return UploadResponse(
            message=f"Successfully processed {url}",
            chunks_count=len(chunks),
            embeddings_count=len(embeddings)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Ask question endpoint
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get an answer based on loaded documents
    """
    try:
        if vector_store.get_size() == 0:
            raise ValueError("No documents loaded. Please upload documents first.")
        
        answer, sources = rag_system.answer_query(request.question)
        
        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Clear vector store endpoint
@app.delete("/clear")
def clear_vectorstore():
    """Clear all documents from vector store"""
    vector_store.clear()
    return {"message": "Vector store cleared"}


# Get stats endpoint
@app.get("/stats")
def get_stats():
    """Get system statistics"""
    return {
        "total_embeddings": vector_store.get_size(),
        "vectorstore_path": VECTORSTORE_PATH,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }


if __name__ == "__main__":
    host = os.getenv('BACKEND_HOST', '127.0.0.1')
    port = int(os.getenv('BACKEND_PORT', 8000))
    uvicorn.run(app, host=host, port=port)
