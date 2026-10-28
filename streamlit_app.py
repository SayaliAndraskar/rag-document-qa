
import os

import faiss
import fitz
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Load the API key from .env
load_dotenv()

st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF Question Answering Assistant")
st.write("Upload a PDF and ask questions based on its contents.")

# ---------- Load models ----------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------- Extract text from PDF ----------

def extract_pdf_text(pdf_bytes):
    text_pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text().strip()

            if page_text:
                text_pages.append(
                    f"[Page {page_number}]\n{page_text}"
                )

    return "\n\n".join(text_pages)


# ---------- Split text into chunks ----------

def split_into_chunks(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------- Build FAISS index ----------

def build_faiss_index(chunks, embedding_model):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# ---------- Process uploaded PDF ----------

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    # Process again when a different PDF is uploaded.
    import hashlib

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        st.session_state.pop("chunks", None)
        st.session_state.pop("faiss_index", None)
        st.session_state["file_hash"] = file_hash

    if "chunks" not in st.session_state:
        with st.spinner("Reading PDF and creating embeddings..."):
            try:
                pdf_text = extract_pdf_text(pdf_bytes)

                if not pdf_text.strip():
                    st.error(
                        "No selectable text was found in this PDF. "
                        "It may be scanned and require OCR."
                    )
                    st.stop()

                chunks = split_into_chunks(pdf_text)

                if not chunks:
                    st.error("Could not create text chunks from this PDF.")
                    st.stop()

                embedding_model = load_embedding_model()
                faiss_index = build_faiss_index(
                    chunks, embedding_model
                )

                st.session_state["chunks"] = chunks
                st.session_state["faiss_index"] = faiss_index

            except Exception as error:
                st.error(f"Could not process the PDF: {error}")
                st.stop()

    st.success(f"PDF ready: {uploaded_file.name}")
    st.caption(
        f"Created {len(st.session_state['chunks'])} text chunks."
    )

    # ---------- Ask questions ----------

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

        if not api_key:
            st.error(
                "GEMINI_API_KEY was not found. "
                "Check that your .env file is in the project folder."
            )
            st.stop()

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding information and generating an answer..."):
                try:
                    embedding_model = load_embedding_model()

                    question_embedding = embedding_model.encode(
                        [question],
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).astype("float32")

                    distances, indices = st.session_state[
                        "faiss_index"
                    ].search(
                        question_embedding,
                        min(4, len(st.session_state["chunks"])),
                    )

                    relevant_chunks = [
                        st.session_state["chunks"][i]
                        for i in indices[0]
                        if i >= 0
                    ]

                    context = "\n\n".join(relevant_chunks)

                    client = genai.Client(api_key=api_key)

                    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the PDF context below.
Do not invent facts or add outside information.
If the context does not contain enough information,
say: "I could not find this information in the PDF."
Give a clear, helpful answer.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                    import time

models = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]

response = None

for model in models:
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            break
        except Exception as e:
            error = str(e)

            if "503" in error or "UNAVAILABLE" in error:
                time.sleep(2 ** (attempt + 1))
            else:
                raise

    if response is not None:
        break

if response is None:
    st.error("Gemini is temporarily busy. Please try again in a minute.")
    st.stop()

answer = response.text

                    if answer:
                        st.markdown(answer)
                    else:
                        st.warning(
                            "The model returned an empty answer. Please try again."
                        )

                    with st.expander("View retrieved PDF text"):
                        st.write(context)

                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")
                    st.info(
                        "If Gemini reports a temporary 503 error, "
                        "wait briefly and try again. If it reports a "
                        "model error, we can select another available model."
                    )

else:
    st.info("Upload a PDF above to get started.")
