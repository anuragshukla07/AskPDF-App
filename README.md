# 🔗💬 AskPDF-App

This project is a **PDF query bot** powered by **Language Learning Models (LLMs)**. Using the app, you can upload a PDF file and ask questions about its content. It processes the file using embeddings and FAISS for text search, combined with Groq's LLM model for answering queries.

This app is built using:
- **Streamlit**: An open-source app framework to build and share data apps.
- **LangChain**: A library for building applications powered by language models.
- **Groq LLM Model**: A high-performance model used to query and generate responses based on PDF content.

## Live Project Link
[Live Project Link](https://askpdf-app.streamlit.app/)  

## Features
- **Upload PDF**: Supports uploading PDF files to extract and query content.
- **LLM-powered answers**: Uses HuggingFace Embeddings and Groq LLM for accurate and intelligent responses.
- **Text splitting**: Handles large PDF files by splitting them into manageable text chunks.
- **Efficient querying**: Employs FAISS to efficiently search across the embedded chunks.
- **Rate-limiting & Retry**: Automatically handles errors like rate-limiting and server unavailability by retrying with exponential backoff.
- **Old file cleanup**: Automatically cleans up expired files to manage storage effectively.


## Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/anuragshukla07/AskPDF-App.git
   ```
2. **Set up a virtual environment**:
   ```
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   ```
3. **Install the dependencies**:
   ```
   pip install -r requirements.txt
   ```
4. **Run the app**:
   ```
   streamlit run app.py
   ```

## Usage
- Upload a PDF: Select a PDF file using the file uploader in the sidebar.
- Ask a Question: Type your query related to the uploaded PDF in the input field.
- Receive an Answer: The app will search the content of the PDF and use the Groq LLM to generate a relevant answer.
- File Cleanup: The app automatically deletes expired files after 120 minutes to save space.

## Technologies Used
- Python: Core programming language for backend and Streamlit app.
- Streamlit: Frontend framework for building the app interface.
- LangChain: Used for text splitting and chain handling for LLM queries.
- FAISS: Vector store used to store and retrieve embedded document text.
- HuggingFace Embeddings: Creates embeddings for the text chunks.
- Groq LLM: Handles the question-answering logic using the uploaded PDF content.

## Feedback
Please provide your valuable feedback [here](https://docs.google.com/forms/d/e/1FAIpQLSdElFrQ7l04vFQzAoe3XIyju597pHFKSKohgJ6t66sZinss5g/viewform)
