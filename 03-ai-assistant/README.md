
# AI Assistant

## Overview

This project develops a Retrieval-Augmented Generation (RAG) based AI assistant that can read industrial machine manuals in PDF format and answer technical questions based on retrieved document information.
The application combines semantic search, vector embeddings, and a Large Language Model (LLM).

## Problem Statement

Industrial technical manuals often contain extensive information about machine specifications, maintenance procedures, troubleshooting, and safety guidelines.
Finding relevant information manually can be time-consuming.
This project explores how Generative AI can help technical personnel retrieve information more efficiently.

## Technologies

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- OpenAI API
- Retrieval-Augmented Generation (RAG)

## System Architecture

1. PDF Document Upload
2. Text Extraction
3. Text Chunking
4. Embedding Generation
5. Vector Index Creation
6. Semantic Search
7. LLM Response Generation
8. Source Reference Display

## Main Features

- Upload industrial PDF manuals
- Extract text from PDF documents
- Split documents into smaller chunks
- Generate semantic embeddings
- Retrieve relevant document passages
- Generate context-based AI answers
- Display retrieved source pages

## Installation

Install Python dependencies:

pip install -r requirements.txt

Create a .env file:

OPENAI_API_KEY=your_api_key

## How to Run

Start the application:

streamlit run app.py

Upload a PDF machine manual.

Click Process Document.

Enter a technical question.

Click Ask AI.

## Example Questions

- Why can machine temperature become high?
- What causes unusual machine vibration?
- What maintenance activities are recommended?
- What should be checked when abnormal temperature occurs?

## RAG Components

### Document Loader

Extracts PDF text and divides it into smaller chunks

### Embeddings

Transforms text into numerical vector representations

### FAISS

Retrieves semantically relevant document chunks

### LLM

Generates answers based on retrieved document context

## Limitations

- Answer quality depends on document quality
- Scanned PDFs may require OCR
- LLM responses may contain errors
- API usage may incur costs
- Safety-critical information must be verified against official manufacturer documentation

## Future Improvements

- Support multiple PDF documents
- Add conversational memory
- Improve document citation
- Add OCR for scanned manuals
- Integrate local LLM models
- Develop an industrial maintenance knowledge base

## Author

Nabilah Robbaniyah — Informatics Student
