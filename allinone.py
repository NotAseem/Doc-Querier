import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from PyPDF2 import PdfReader
import os
import shutil
import argparse

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Helper Functions
def process_pdf(file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text):
    """Splits extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    documents = [Document(page_content=text, metadata={"source": "uploaded_file"})]
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks):
    """Add chunks of text to the Chroma database."""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Generate unique IDs for chunks
    for idx, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"uploaded_file_chunk_{idx}"

    # Get existing document IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    st.write(f"Number of existing documents in DB: {len(existing_ids)}")

    # Add only new documents
    new_chunks = [
        chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        st.write(f"Adding {len(new_chunks)} new document(s) to the database.")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        st.write("No new documents to add.")
    return db

def query_rag(query_text: str, db):
    """Query the RAG system and return the response."""
    results = db.similarity_search_with_score(query_text, k=5)

    # Create context from retrieved documents
    context_text = " ".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query the model
    model = ChatOllama(model="llama3.2:3b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return response_text, sources

def clear_database():
    """Clear the Chroma database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        st.write("Database cleared.")

# Streamlit Interface
def main():
    # CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args, unknown = parser.parse_known_args()

    if args.reset:
        clear_database()

    # Streamlit UI
    st.title("PDF Querying with RAG System")

    # Clear Database Option
    if st.button("Clear Database"):
        clear_database()

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("Processing PDF...")
        pdf_text = process_pdf(uploaded_file)
        st.success("PDF uploaded and processed!")

        # Split text into chunks
        st.write("Splitting text into chunks...")
        chunks = split_text_into_chunks(pdf_text)
        st.success(f"Split into {len(chunks)} chunks!")

        # Set up the Chroma database
        st.write("Setting up the database...")
        db = add_to_chroma(chunks)
        st.success("Database populated successfully!")

        # Query Input
        query = st.text_input("Enter your query:")
        if query:
            st.write("Searching for answers...")
            response, sources = query_rag(query, db)

            # Display Results
            st.write("### Response")
            st.write(response.content)

            st.write("### Sources")
            st.write(sources)

if __name__ == "__main__":
    main()