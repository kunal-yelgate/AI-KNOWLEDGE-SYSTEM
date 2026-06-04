"""
Embedding Module
Generate embeddings using Gemini API
"""

import os
from typing import List
import warnings

# Suppress FutureWarning about deprecated package
warnings.filterwarnings('ignore', category=FutureWarning)

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings using Gemini API"""

    def __init__(self, api_key: str = None):
        """Initialize with Gemini API key"""
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        # Try newer model first, fall back to other options
        self.model = "models/text-embedding-004"

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
        
        Returns:
            List of embedding values
        """
        try:
            # Try with newest model first
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="SEMANTIC_SIMILARITY"
                )
            except Exception as e:
                if "not found" in str(e):
                    # Fallback to alternative model
                    self.model = "models/embedding-001"
                    result = genai.embed_content(
                        model=self.model,
                        content=text
                    )
                else:
                    raise
            
            return result['embedding']
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding lists
        """
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings


class QueryEmbedder:
    """Embed user queries"""

    def __init__(self, api_key: str = None):
        """Initialize embedder"""
        self.generator = EmbeddingGenerator(api_key)

    def embed_query(self, query: str) -> List[float]:
        """Embed a user query"""
        return self.generator.generate_embedding(query)
