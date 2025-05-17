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
import uuid

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
    return text, file.name  # Return both text and filename

def split_text_into_chunks(text):
    """Splits extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    documents = [Document(page_content=text)]
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks, filename):
    """Add chunks of text to the Chroma database."""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Generate unique IDs for chunks using UUID
    unique_id = str(uuid.uuid4())
    for idx, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"{filename}_{unique_id}_chunk_{idx}"
        chunk.metadata["source"] = filename

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

    # Query the model with streaming
    model = ChatOllama(model="llama3.2:3b", streaming=True)
    response_text = ""
    for chunk in model.stream(prompt):
        if hasattr(chunk, 'content') and chunk.content:
            response_text += chunk.content
            yield chunk.content, None  # Yield each chunk as it's generated

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    yield None, sources  # Yield sources at the end

# def clear_database():
#     """Clear the Chroma database."""
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)
#         st.write("Database cleared.")

# Streamlit Interface
def main():
    # CLI Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args, unknown = parser.parse_known_args()
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    if args.reset:
        # clear_database()
        pass

    # Streamlit UI

    st.title("PDF Querying with RAG System")

    # Clear Database Option
    # if st.button("Clear Database"):
    #     clear_database()

    existing_items = db.get(include=[])
    has_documents = len(existing_items["ids"]) > 0
    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("Processing PDF...")
        pdf_text, filename = process_pdf(uploaded_file)
        st.success(f"PDF '{filename}' uploaded and processed!")

        # Split text into chunks
        st.write("Splitting text into chunks...")
        chunks = split_text_into_chunks(pdf_text)
        st.success(f"Split into {len(chunks)} chunks!")

        # Set up the Chroma database
        st.write("Setting up the database...")
        db = add_to_chroma(chunks, filename)
        st.success("Database populated successfully!")

        # Query Input
        #query = st.text_input("Enter your query:")
        # if query:
        #     st.write("Searching for answers...")
        #     response, sources = query_rag(query, db)

        #     # Display Results
        #     st.write("### Response")
        #     st.write(response.content)

        #     st.write("### Sources")
        #     st.write(sources)
    if has_documents or uploaded_file is not None:
        st.subheader("📂 Files in Database")
    
        # --- Show existing documents ---
        try:
            # Get all documents and their metadata
            all_docs = db.get()
            if all_docs and all_docs['metadatas']:
                # Create a set of unique filenames from the source metadata
                unique_files = set()
                for metadata in all_docs['metadatas']:
                    if 'source' in metadata:
                        unique_files.add(metadata['source'])
                
                # Display each unique filename
                for filename in unique_files:
                    st.markdown(f"- 📄 `{filename}`")
            else:
                st.info("No files found in the database.")
        except Exception as e:
            st.error(f"Error fetching files: {e}")

        st.subheader("🔍 Ask a Question")
        query = st.text_input("Enter your query:")
        if query:
            with st.spinner("🔍 Searching through documents..."):
                response_stream = query_rag(query, db)
                
                # Create an empty container for the response
                response_container = st.empty()
                full_response = ""
                
                # Stream the response
                for chunk, sources in response_stream:
                    if chunk is not None:
                        full_response += chunk
                        response_container.markdown(full_response)
    else:
        st.info("Upload a PDF to enable querying.")

if __name__ == "__main__":
    main()