"""
Streamlit Frontend for RAG System
Simple web UI for uploading documents and asking questions
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 RAG Chatbot - Powered by Gemini")
st.markdown("Upload documents or URLs, then ask questions based on their content.")

# Sidebar for document management
with st.sidebar:
    st.header("📚 Document Management")
    
    # Check API health
    try:
        health = requests.get(f"{API_BASE_URL}/health").json()
        st.success(f"✅ Backend Connected | {health['vectorstore_size']} embeddings loaded")
    except:
        st.error("❌ Backend not running. Start it with: uvicorn backend.app:app --reload")
        st.stop()
    
    # Upload file section
    st.subheader("Upload File")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    
    if uploaded_file:
        if st.button("📤 Upload & Process", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getbuffer())}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    result = response.json()
                    
                    st.success(f"✅ {result['message']}")
                    st.info(f"Chunks: {result['chunks_count']} | Embeddings: {result['embeddings_count']}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Upload from URL section
    st.subheader("Load from URL")
    url_input = st.text_input("Enter website URL or article link")
    
    if st.button("🌐 Load from URL", use_container_width=True):
        if url_input:
            with st.spinner("Processing URL..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/upload-url",
                        params={"url": url_input}
                    )
                    result = response.json()
                    
                    st.success(f"✅ {result['message']}")
                    st.info(f"Chunks: {result['chunks_count']} | Embeddings: {result['embeddings_count']}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Statistics
    st.subheader("📊 Statistics")
    try:
        stats = requests.get(f"{API_BASE_URL}/stats").json()
        st.metric("Total Embeddings", stats["total_embeddings"])
        st.metric("Chunk Size", stats["chunk_size"])
    except:
        pass
    
    # Clear button
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        try:
            requests.delete(f"{API_BASE_URL}/clear")
            st.success("Vector store cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")


# Main chat interface
st.subheader("💬 Ask Questions")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Question input
question = st.chat_input("Ask a question about the uploaded documents...")

if question:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    # Get answer from backend
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": question}
            )
            result = response.json()
            answer = result["answer"]
            sources = result["sources"]
            
            # Add assistant message to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            # Display answer
            st.chat_message("assistant").write(answer)
            
            # Display sources
            with st.expander("📖 Sources"):
                for i, source in enumerate(sources, 1):
                    st.text_area(f"Source {i}:", value=source, disabled=True, height=100)
        
        except Exception as e:
            st.error(f"Error getting answer: {str(e)}")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>RAG System powered by <b>Gemini AI</b> | Built with Streamlit & FastAPI</p>
    </div>
    """,
    unsafe_allow_html=True
)
