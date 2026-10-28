
import os
import time
import hashlib

import streamlit as st
import fitz
import faiss
import numpy as np

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Load local .env file
load_dotenv()

st.set_page_config(
    page_title="RAG PDF Question Answering",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Question Answering Assistant")
st.write("Upload a PDF and ask questions based on its contents.")


# Load embedding model only once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# Read PDF
def extract_pdf_text(pdf_bytes):
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

    return "\n\n".join(pages)


# Split text into chunks
def split_into_chunks(text, chunk_size=1000, overlap=150):
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# Create FAISS index
def build_faiss_index(chunks, model):
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# Get API key from environment or Streamlit Cloud Secrets
def get_api_key():
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None

    return key


# Generate answer with retries and fallback models
def generate_answer(client, prompt):
    models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash"
    ]

    last_error = None

    for model_name in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if response.text:
                    return response.text, model_name

                last_error = "The model returned an empty answer."
                break

            except Exception as error:
                last_error = error
                error_text = str(error).upper()

                temporary_error = any(
                    term in error_text
                    for term in [
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "500",
                        "INTERNAL"
                    ]
                )

                if temporary_error and attempt == 0:
                    time.sleep(2)
                    continue

                if temporary_error:
                    break

                raise

    raise RuntimeError(
        "Gemini models are temporarily unavailable or the API quota "
        f"may have been reached. Last error: {last_error}"
    )


# PDF upload
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("Upload a PDF above to get started.")
    st.stop()


# Process a new PDF only when its contents change
pdf_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("file_hash") != file_hash:
    st.session_state["file_hash"] = file_hash
    st.session_state.pop("chunks", None)
    st.session_state.pop("faiss_index", None)
    st.session_state.pop("messages", None)

if "chunks" not in st.session_state:
    with st.spinner("Reading PDF and creating embeddings..."):
        try:
            pdf_text = extract_pdf_text(pdf_bytes)

            if not pdf_text:
                st.error(
                    "No selectable text found. This PDF may be scanned "
                    "and require OCR."
                )
                st.stop()

            chunks = split_into_chunks(pdf_text)

            embedding_model = load_embedding_model()
            index = build_faiss_index(chunks, embedding_model)

            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = index

        except Exception as error:
            st.error(f"Could not process PDF: {error}")
            st.stop()

st.success(f"PDF ready: {uploaded_file.name}")
st.caption(f"Created {len(st.session_state['chunks'])} text chunks.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Show previous conversation
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Ask questions
question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding relevant text and generating answer..."):
            try:
                api_key = get_api_key()

                if not api_key:
                    st.error(
                        "GEMINI_API_KEY not found. Add it under "
                        "Streamlit Cloud Settings → Secrets."
                    )
                    st.stop()

                embedding_model = load_embedding_model()

                question_embedding = embedding_model.encode(
                    [question],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).astype("float32")

                number_to_retrieve = min(
                    4, len(st.session_state["chunks"])
                )

                distances, indices = st.session_state[
                    "faiss_index"
                ].search(question_embedding, number_to_retrieve)

                relevant_chunks = [
                    st.session_state["chunks"][i]
                    for i in indices[0]
                    if i >= 0
                ]

                context = "\n\n".join(relevant_chunks)

                prompt = f"""
You are a PDF question-answering assistant.

Answer using only the provided PDF context.
Do not invent facts or add outside information.
If the answer is not present, say:
"I could not find this information in the PDF."
Keep the answer clear and helpful.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                client = genai.Client(api_key=api_key)
                answer, model_used = generate_answer(client, prompt)

                st.markdown(answer)
                st.caption(f"Model used: {model_used}")

                with st.expander("View retrieved PDF text"):
                    st.write(context)

                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(f"Could not generate an answer: {error}")
                st.info(
                    "If the service is busy, wait a little and retry. "
                    "Repeated failures may indicate an API quota or "
                    "model availability issue."
                )

