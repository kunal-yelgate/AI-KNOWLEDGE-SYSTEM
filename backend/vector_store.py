"""
Vector Store Module
FAISS-based vector database for storing and retrieving embeddings
"""

import os
import pickle
import numpy as np
from typing import List, Tuple
import faiss


class FAISSVectorStore:
    """FAISS-based vector database"""

    def __init__(self, dimension: int = 768, store_path: str = "./vectorstore"):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension (Gemini text-embedding-004 uses 768)
            store_path: Path to store index and metadata
        """
        self.dimension = dimension
        self.store_path = store_path
        self.index_path = os.path.join(store_path, "index.faiss")
        self.metadata_path = os.path.join(store_path, "metadata.pkl")
        
        # Create store directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)
        
        # Initialize or load index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = []

    def add_embeddings(self, embeddings: List[List[float]], texts: List[str], 
                      metadata: List[dict] = None):
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: List of embedding vectors
            texts: List of corresponding text chunks
            metadata: Optional metadata for each text
        """
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store metadata
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            self.metadata.append({
                'text': text,
                'metadata': meta
            })
        
        # Save index and metadata
        self.save()

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of top results to return
        
        Returns:
            List of (text, distance) tuples
        """
        query_array = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.index.search(query_array, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                text = self.metadata[idx]['text']
                results.append((text, float(distance)))
        
        return results

    def save(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """Load index and metadata from disk"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

    def clear(self):
        """Clear the vector store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.save()

    def get_size(self) -> int:
        """Get number of embeddings in the store"""
        return self.index.ntotal
