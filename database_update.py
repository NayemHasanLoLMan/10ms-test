import os
import time
import google.generativeai as genai
import fitz  # PyMuPDF for PDF extraction
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

import math
from textwrap import wrap

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client using the new way
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define Serverless specifications
serverless_spec = ServerlessSpec(
    cloud="aws",       
    region="us-east-1"  
)

# Index name
index_name = "hsc26-bangla-1st-paper"  # Name of the index

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension based on Gemini embedding
        metric="cosine",  # metric for embeddings
        spec=serverless_spec 
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to extract text from a PDF file
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)  
    pdf_text = []
    
    # Extract text page by page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        text = page.get_text()  # Extract text from page
        
        if text.strip():  # Check if the text is not empty or just whitespace
            pdf_text.append(text)
        else:
            pdf_text.append('')  # Append empty string for pages with no text
    
    return pdf_text, doc.metadata.get("title", "Unknown Title")  

# Function to embed text using Gemini's `text-embedding-004` model
def embed_text_with_gemini(text):
    if text.strip():  # Ensure the text is not empty or whitespace
        response = genai.embed_content(
            model='text-embedding-004',
            content=text,
            task_type="retrieval_document"  
        )
        return response['embedding']
    else:
        print("Warning: Skipping empty page.")
        return None  


# Function to chunk text into max_length
def chunk_text(text, max_length=8000):
    return wrap(text, width=max_length, break_long_words=False, replace_whitespace=False)


def upload_pdf_to_pinecone(pdf_path):
    pdf_text, pdf_title = extract_pdf_text(pdf_path)
    
    for page_num, text in enumerate(pdf_text):
        if not text.strip():
            print(f"Skipping Page {page_num + 1} as it contains no valid text.")
            continue

        # Split long text into multiple chunks
        chunks = chunk_text(text, max_length=8000)

        for chunk_index, chunk in enumerate(chunks):
            embedding_vector = embed_text_with_gemini(chunk)
            
            if embedding_vector is not None:
                # Unique ID per chunk (even if same page)
                page_id = f"page-{page_num + 1}-chunk-{chunk_index + 1}"
                
                metadata = {
                    "pdf_title": pdf_title,
                    "page_number": page_num + 1,
                    "chunk_number": chunk_index + 1,
                    "text": chunk,
                    "char_count": len(chunk)
                }
                
                index.upsert([
                    (page_id, embedding_vector, metadata)
                ])
                print(f"Page {page_num + 1} Chunk {chunk_index + 1} embedded and upserted.")
            else:
                print(f"Skipping Page {page_num + 1} Chunk {chunk_index + 1} due to empty content.")


# Uploading a PDF to Pinecone
pdf_path = "C:\\Users\\hasan\\Downloads\\10ms Assesment\\HSC26-Bangla1st-Paper.pdf"  
upload_pdf_to_pinecone(pdf_path)

# Success message
print("All pages from the PDF have been successfully embedded and upserted into Pinecone.")