"""
Data Loader Module
Handles loading data from PDFs, text files, and web URLs
"""

import os
import re
import requests
from typing import List, Tuple
from urllib.parse import urlparse
from pypdf import PdfReader
from bs4 import BeautifulSoup


class DataLoader:
    """Load and clean data from various sources"""

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def load_text_file(file_path: str) -> str:
        """Load text from .txt file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_web_url(url: str) -> str:
        """Scrape and extract text from web URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            return text
        
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch URL: {url}. Error: {str(e)}")

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\'\"]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    @staticmethod
    def load_document(source: str) -> Tuple[str, str]:
        """
        Load document from any source (PDF, TXT, or URL)
        Returns: (cleaned_text, source_type)
        """
        source = source.strip()
        
        # Check if it's a URL
        if source.startswith('http://') or source.startswith('https://'):
            text = DataLoader.load_web_url(source)
            source_type = "web"
        
        # Check if it's a file path
        elif os.path.exists(source):
            if source.lower().endswith('.pdf'):
                text = DataLoader.load_pdf(source)
                source_type = "pdf"
            elif source.lower().endswith('.txt'):
                text = DataLoader.load_text_file(source)
                source_type = "text"
            else:
                raise ValueError(f"Unsupported file format: {source}")
        
        else:
            raise ValueError(f"Invalid source (not a file or URL): {source}")
        
        # Clean the text
        cleaned_text = DataLoader.clean_text(text)
        return cleaned_text, source_type


class TextChunker:
    """Split text into overlapping chunks"""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to split
            chunk_size: Size of each chunk (in tokens/words)
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    sent_len = len(sent.split())
                    if overlap_length + sent_len <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += sent_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
