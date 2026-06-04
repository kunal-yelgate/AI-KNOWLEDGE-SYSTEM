"""
Retrieval and Answer Generation Module
Combine retrieved context with Gemini to generate answers
"""

import os
from typing import List, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class RAGSystem:
    """Retrieval-Augmented Generation System"""

    def __init__(self, vector_store, embedding_generator, api_key: str = None):
        """
        Initialize RAG system
        
        Args:
            vector_store: FAISSVectorStore instance
            embedding_generator: EmbeddingGenerator instance
            api_key: Gemini API key
        """
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.model = genai.GenerativeModel('gemini-pro')
        self.top_k = int(os.getenv('TOP_K_RESULTS', 3))

    def retrieve_context(self, query: str) -> List[str]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
        
        Returns:
            List of relevant text chunks
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=self.top_k)
        
        # Extract texts
        context_texts = [text for text, _ in results]
        return context_texts

    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generate answer using Gemini with retrieved context
        
        Args:
            query: User query
            context: List of relevant context chunks
        
        Returns:
            Generated answer
        """
        # Format context
        context_text = "\n---\n".join(context)
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the question. 
If the answer is not in the context, say "I don't have enough information in the provided context to answer this question."

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {str(e)}")

    def answer_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Full RAG pipeline: retrieve + generate
        
        Args:
            query: User query
        
        Returns:
            Tuple of (answer, context_sources)
        """
        # Retrieve context
        context = self.retrieve_context(query)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        return answer, context
