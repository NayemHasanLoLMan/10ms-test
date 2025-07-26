import os
import re
import time
from dotenv import load_dotenv
from docx import Document
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "bangla-docx-embeddings"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    # Join all paragraphs into one long string
    text = " ".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    return text

def clean_bangla_text(text):
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'\\u0000|\u0000|\\[a-zA-Z0-9]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u0980-\u09FF\u0020-\u007E\s।,.:;!?()-]', '', text)
    text = re.sub(r'[।]{2,}', '।', text)
    text = re.sub(r'[,]{2,}', ',', text)
    return text.strip()

def create_overlapping_chunks(text, chunk_size=1500, overlap=500):
    """
    Create chunks of the specified size (chunk_size) with overlap.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end > text_len:
            end = text_len
        else:
            # Ensure overlap
            end = min(end, text_len)

        chunk = text[start:end].strip()

        # If chunk is very small and there is a previous chunk, merge it to previous chunk
        if len(chunk) < 300 and chunks:
            # Append small chunk to previous chunk with a space separator
            chunks[-1] += " " + chunk
        else:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Move start forward with overlap
        start = end - overlap

    return chunks

def embed_with_gemini(text, retries=5, delay=15):
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model='gemini-embedding-001',
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            print(f"⚠️ Retry {attempt + 1}/{retries} - Error creating embedding: {e}")
            time.sleep(delay)
    return None

def upload_docx_to_pinecone(docx_path):
    print(f"📄 Processing DOCX: {docx_path}")
    
    raw_text = extract_text_from_docx(docx_path)
    cleaned_text = clean_bangla_text(raw_text)
    
    if not cleaned_text:
        print("❌ No valid text found.")
        return 0

    # Create chunks based on character length
    chunks = create_overlapping_chunks(cleaned_text, chunk_size=2500, overlap=500)

    total_chunks_uploaded = 0

    for chunk_index, chunk in enumerate(chunks):
        if len(chunk) < 50:
            # Skip super tiny chunks if any remain
            continue

        embedding = embed_with_gemini(chunk)

        if embedding:
            vector_id = f"chunk_{chunk_index + 1}"
            metadata = {
                "source": os.path.basename(docx_path),
                "chunk_index": chunk_index + 1,
                "text": chunk,
                "char_count": len(chunk),
                "original_text": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
            }

            index.upsert([(vector_id, embedding, metadata)])
            total_chunks_uploaded += 1

            print(f"✅ Uploaded: Chunk {chunk_index + 1} ({len(chunk)} chars)")
        else:
            print(f"❌ Failed to embed: Chunk {chunk_index + 1}")

    print(f"\n🎉 Total chunks uploaded: {total_chunks_uploaded}")
    return total_chunks_uploaded

# Run this script
if __name__ == "__main__":
    docx_file_path = "C:\\Users\\hasan\\Assesment\\extracted_text_bangla.docx"  # Update as needed
    upload_docx_to_pinecone(docx_file_path)
