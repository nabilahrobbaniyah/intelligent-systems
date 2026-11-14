# AI ASSISTANT, RAG APPLICATION

import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from src.document_loader import (load_pdf, create_chunks)

from src.embeddings import (
    create_embedding_model,
    build_vector_index,
    search_similar_chunks
)

from src.rag_engine import generate_answer

# 1. CONFIGURATION
load_dotenv()

st.set_page_config(
    page_title="Industrial AI Assistant",
    layout="wide"
)

st.title("Industrial AI Assistant")

st.write(
    "Ask questions about industrial machine manuals "
    "using Retrieval-Augmented Generation (RAG)."
)

# 2. SESSION STATE
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

# 3. SIDEBAR
st.sidebar.header("Document Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Machine Manual (PDF)",
    type=["pdf"]
)

api_key = os.getenv("OPENAI_API_KEY", "")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=api_key,
    type="password",
    help="Your API key is used to generate answers."
)

top_k = st.sidebar.slider(
    "Retrieved Chunks",
    min_value=1,
    max_value=5,
    value=3
)

# 4. DOCUMENT PROCESSING
if st.button("Process Document", type="primary"):

    if uploaded_file is None:
        st.warning("Please upload a PDF document.")
    else:

        try:
            with st.spinner("Reading and processing document..."):

                # Save temporary PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    # Extract PDF text
                    documents = load_pdf(temp_path)

                finally:
                    os.remove(temp_path)

                if not documents:
                    st.error("No readable text found in PDF")
                    st.stop()

                # Split document into chunks
                chunks = create_chunks(documents)

                if not chunks:
                    st.error("No text chunks generated.")
                    st.stop()

                # Load embedding model
                embedding_model = create_embedding_model()

                # Build FAISS index
                index = build_vector_index(chunks, embedding_model)

                # Save to session state
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.embedding_model = embedding_model
                st.session_state.document_name = (uploaded_file.name)

            st.success("Document processed successfully!")
            st.write(f"Document: {uploaded_file.name}")
            st.write(f"Pages extracted: {len(documents)}")
            st.write(f"Text chunks created: {len(chunks)}")

        except Exception as error:
            st.error(f"Document processing failed: {error}")



# 5. DOCUMENT STATUS
if st.session_state.chunks is not None:
    st.divider()

    st.success(
        f"Active document: "
        f"{st.session_state.document_name}"
    )

    st.write(
        f"Available chunks: "
        f"{len(st.session_state.chunks)}"
    )

# 6. QUESTION INPUT
st.subheader("Ask Your Industrial Assistant")
question = st.text_input("Enter your question", placeholder="Why can machine temperature become high?")

if st.button("Ask AI"):

    if st.session_state.index is None:
        st.warning("Please process a PDF document first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    elif not api_key.strip():
        st.warning("Please provide your OpenAI API key.")

    else:
        try:
            with st.spinner("Searching document and generating answer..."):

                # Retrieve relevant information
                retrieved_chunks = search_similar_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.embedding_model,
                    st.session_state.index,
                    top_k=top_k
                )

                # Generate answer
                answer = generate_answer(question, retrieved_chunks)

            # Display answer
            st.subheader("AI Answer")
            st.markdown(answer)

            # Display references
            st.subheader("Retrieved Sources")

            for i, chunk in enumerate(retrieved_chunks, start=1):
                with st.expander(f"Source {i} - Page {chunk['page']}"):
                    st.write(chunk["text"])

                    st.caption(
                        f"Similarity score: "
                        f"{chunk['score']:.4f}"
                    )

        except Exception as error:
            st.error(f"An error occurred: {error}")

# 7. FOOTER
st.divider()

st.caption(
    "Educational prototype. Verify all technical "
    "and safety information against official manuals."
)
