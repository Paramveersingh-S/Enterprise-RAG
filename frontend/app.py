import os
import json
import streamlit as st
import requests
import time

# Use the internal docker network API endpoint
API_URL = os.getenv("API_URL", "http://api:8000/api/v1")

st.set_page_config(
    page_title="Enterprise Graph RAG",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ Enterprise Graph RAG")
st.markdown("A highly sophisticated **Hybrid Graph RAG** engine combining Dense Vector Search, Sparse BM25 Retrieval, and Knowledge Graphs.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Ingestion
with st.sidebar:
    st.header("📄 Document Ingestion")
    st.markdown("Upload documents to build the Knowledge Graph and Vector Index.")
    
    # Actually saving the uploaded file into the /app/data directory so the worker can read it
    uploaded_file = st.file_uploader("Upload a file (PDF, MD, TXT, DOCX)", type=["pdf", "md", "txt", "docx"])
    
    if st.button("Ingest Document") and uploaded_file is not None:
        with st.spinner("Uploading to server..."):
            # Ensure data dir exists
            os.makedirs("/app/data", exist_ok=True)
            file_path = f"/app/data/{uploaded_file.name}"
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Call FastAPI Ingest endpoint
            try:
                response = requests.post(
                    f"{API_URL}/ingest",
                    json={"file_paths": [file_path]},
                    timeout=10
                )
                if response.status_code == 200:
                    task_id = response.json().get("task_id")
                    st.success(f"Ingestion started! Task ID: `{task_id}`")
                    st.info("Check `docker-compose logs -f worker` for progress.")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What are the key terms mentioned in the document?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Searching Knowledge Graph & Vector DB..."):
                response = requests.post(
                    f"{API_URL}/query_stream",
                    json={"query": prompt},
                    stream=True,
                    timeout=60
                )
                
            if response.status_code == 200:
                answer = ""
                metrics_html = ""
                
                # Streamlit component to update the text as it streams
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("type") == "token":
                            answer += data["content"]
                            message_placeholder.markdown(answer + "▌")
                        elif data.get("type") == "metrics":
                            # Final stream payload contains metrics
                            relevance = data.get("relevance_score", 0.0)
                            hallucination = data.get("hallucination_score", 0.0)
                            sources = data.get("sources", [])
                            
                            metrics_md = f"\n\n---\n**Metrics:**\n"
                            metrics_md += f"- Relevance Score: `{relevance:.2f}`\n"
                            metrics_md += f"- Hallucination Score: `{hallucination:.2f}`\n"
                            if sources:
                                metrics_md += f"- Sources: {', '.join([f'`{s[:8]}...`' for s in sources])}\n"
                            
                            # Final output without the cursor
                            message_placeholder.markdown(answer + metrics_md)
                            st.session_state.messages.append({"role": "assistant", "content": answer + metrics_md})
                        elif data.get("type") == "error":
                            st.error(f"Streaming Error: {data.get('content')}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
