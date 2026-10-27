# RAG-Based PDF Question Answering Assistant

A Python-based web application that allows users to upload PDF documents and ask questions about their content.

## Features
- PDF text extraction using PyMuPDF
- Text embeddings using Sentence Transformers
- Semantic search using FAISS
- Context-based answers using Google Gemini API
- Interactive interface built with Streamlit

## Technologies
Python | Streamlit | FAISS | Sentence Transformers | Google Gemini API | PyMuPDF

## Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file and add your `GEMINI_API_KEY`.
3. Run: `python -m streamlit run streamlit_app.py`

Created by Sayali.
