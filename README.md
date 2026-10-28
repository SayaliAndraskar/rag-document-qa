# PDF Question Answering Assistant

An AI-powered web application that allows users to upload a PDF and ask questions about its content using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

[**Try the PDF Question Answering Assistant**](https://rag-document-app-fvk7azrghtu6sug9pappwex.streamlit.app)

## ✨ Features

- Upload PDF documents.
- Extract text using PyMuPDF.
- Generate text embeddings using Sentence Transformers.
- Retrieve relevant document chunks using FAISS.
- Generate answers using the Gemini API.
- Interactive web interface built with Streamlit.

## 🛠️ Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Google Gemini API

## ⚙️ How It Works

1. The user uploads a PDF.
2. The application extracts text from the PDF.
3. Text is divided into chunks and converted into embeddings.
4. FAISS retrieves chunks relevant to the user's question.
5. Gemini generates an answer using the retrieved context.
6. The answer is displayed in the web interface.

## 💻 Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/SayaliAndraskar/rag-document-qa.git
   ```

2. Open the project folder:

   ```bash
   cd rag-document-qa
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure your Gemini API key using an environment variable or a local `.env` file. Never commit your API key to GitHub.

5. Run the application:

   ```bash
   streamlit run streamlit_app.py
   ```

## 👩‍💻 Author

Sayali Andraskar

## 📌 GitHub Repository

[View Source Code](https://github.com/SayaliAndraskar/rag-document-qa)
