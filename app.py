
import os
import fitz
import faiss
import numpy as np

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ==========================================
# STEP 1: LOAD GEMINI API KEY
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)


# ==========================================
# STEP 2: LOAD AND EXTRACT PDF TEXT
# ==========================================

PDF_PATH = "sample.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        "sample.pdf not found. Put it in the project folder."
    )

print("\nOpening PDF...")

pdf = fitz.open(PDF_PATH)
pages_text = []

for page_number, page in enumerate(pdf, start=1):
    page_text = page.get_text("text")
    if page_text.strip():
        pages_text.append(page_text)

pdf.close()

full_text = "\n".join(pages_text)

print("PDF pages with readable text:", len(pages_text))
print("Extracted text length:", len(full_text))

if not full_text.strip():
    raise ValueError(
        "No readable text found. This PDF may be scanned."
    )

print("\nFirst 300 characters of extracted text:")
print(full_text[:300])


# ==========================================
# STEP 3: SPLIT TEXT INTO CHUNKS
# ==========================================

chunk_size = 900
overlap = 150
chunks = []

start = 0

while start < len(full_text):
    chunk = full_text[start:start + chunk_size]

    if chunk.strip():
        chunks.append(chunk)

    start += chunk_size - overlap

print("\nTotal chunks created:", len(chunks))

if not chunks:
    raise ValueError("No text chunks were created.")


# ==========================================
# STEP 4: CREATE TEXT EMBEDDINGS
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

print("Embedding shape:", embeddings.shape)


# ==========================================
# STEP 5: CREATE FAISS SEARCH INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("\nPDF is ready for questions!")


# ==========================================
# STEP 6: ASK QUESTIONS
# ==========================================

while True:
    question = input(
        "\nAsk a question about the PDF "
        "(or type exit): "
    ).strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    # ======================================
    # STEP 7: RETRIEVE RELEVANT TEXT
    # ======================================

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(5, len(chunks))

    scores, indices = index.search(
        question_embedding, k
    )

    retrieved_chunks = []

    for chunk_index in indices[0]:
        if 0 <= chunk_index < len(chunks):
            retrieved_chunks.append(chunks[chunk_index])

    context = "\n\n".join(retrieved_chunks).strip()

    print("\n--- Retrieved context ---")

    if context:
        print(context)
    else:
        print("No text was retrieved.")

    print("--- End of retrieved context ---")

    if not context:
        print(
            "Retrieval failed. Check the PDF and extracted text."
        )
        continue

    # ======================================
    # STEP 8: GENERATE ANSWER WITH GEMINI
    # ======================================

    prompt = f"""
You are a document question-answering assistant.

Answer the question using the provided PDF context.
Do not invent facts or use outside information.
If the context does not contain the answer, say:
"I could not find this information in the PDF."

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        print("\nGenerating answer...")

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        answer = response.text

        if answer:
            print("\nAnswer:")
            print(answer)
        else:
            print("Gemini returned an empty answer.")

    except Exception as error:
        print("\nGemini request failed:")
        print(error)


    