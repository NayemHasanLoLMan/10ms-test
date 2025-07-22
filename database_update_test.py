# import random
# import string
# import fitz  # PyMuPDF
# from typing import List
# import openai
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# import os
# import re

# load_dotenv()

# class WordVectorizerOpenAIPinecone:
#     def __init__(self, pinecone_index_name: str):
#         self.index_name = pinecone_index_name
#         # FIX 1: Use newer embedding model that works better with Bengali
#         self.embedding_model = "text-embedding-3-small"
#         openai.api_key = os.getenv("OPENAI_API_KEY")

#         self.pc = Pinecone(os.environ.get("PINECONE_API_KEY"))
#         if pinecone_index_name not in [i.name for i in self.pc.list_indexes()]:
#             print(f"Creating new index: {pinecone_index_name}")
#             self.pc.create_index(
#                 name=pinecone_index_name,
#                 dimension=1536,  # text-embedding-3-small dimension
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud="aws", region="us-east-1")
#             )

#         self.index = self.pc.Index(pinecone_index_name)

#     def generate_unique_id(self) -> str:
#         return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

#     def sanitize_vector_id(self, vector_id: str) -> str:
#         # FIX 2: Better sanitization that preserves Bengali characters
#         sanitized = re.sub(r'[^\w\d\u0980-\u09FF]', '_', vector_id)
#         sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
#         if len(sanitized) > 64:
#             prefix = sanitized[:50]
#             suffix = self.generate_unique_id()
#             sanitized = f"{prefix}_{suffix}"
#         return sanitized

#     def clean_text(self, text: str) -> str:
#         """Improved cleaning for Bengali text."""
#         if not text or len(text.strip()) < 20:
#             return ""
            
#         # Remove excessive whitespace but preserve structure
#         text = re.sub(r'\n+', '\n', text)
#         text = re.sub(r'[ \t]+', ' ', text)
#         text = text.strip()
        
#         # Remove page numbers in both Bengali and English
#         text = re.sub(r'(পৃষ্ঠা\s*\d+|Page\s*\d+|পাতা\s*\d+)', '', text, flags=re.IGNORECASE)
        
#         # Remove PDF artifacts
#         text = re.sub(r'[\x0c\x00-\x08\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
#         # Clean up excessive punctuation
#         text = re.sub(r'[।]{2,}', '।', text)
        
#         return text.strip()

#     def create_embedding(self, text: str) -> List[float]:
#         """FIX 3: Use new OpenAI client and better chunking."""
#         try:
#             if not text or len(text.strip()) < 10:
#                 return []
            
#             # Use the new OpenAI client
#             from openai import OpenAI
#             client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
#             # FIX 4: Better chunking strategy for long text
#             max_chunk_size = 8000  # Conservative limit
#             if len(text) <= max_chunk_size:
#                 # Single chunk
#                 response = client.embeddings.create(
#                     model=self.embedding_model,
#                     input=text,
#                     encoding_format="float"
#                 )
#                 return response.data[0].embedding
#             else:
#                 # Multiple chunks with overlap
#                 chunks = []
#                 overlap = 200
#                 for i in range(0, len(text), max_chunk_size - overlap):
#                     chunk = text[i:i + max_chunk_size]
#                     if chunk.strip():
#                         chunks.append(chunk.strip())
#                     if i + max_chunk_size >= len(text):
#                         break

#                 if not chunks:
#                     return []

#                 embeddings = []
#                 for chunk in chunks:
#                     response = client.embeddings.create(
#                         model=self.embedding_model,
#                         input=chunk,
#                         encoding_format="float"
#                     )
#                     embeddings.append(response.data[0].embedding)

#                 # Average the embeddings
#                 if embeddings:
#                     averaged_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
#                     return averaged_embedding
#                 else:
#                     return []

#         except Exception as e:
#             print(f"Error creating embedding: {e}")
#             return []

#     def extract_clean_text_from_pdf_page(self, doc: fitz.Document, page_num: int) -> str:
#         """FIX 5: Better text extraction for Bengali PDFs."""
#         try:
#             page = doc.load_page(page_num)
            
#             # Try structured extraction first (better for Bengali)
#             text_dict = page.get_text("dict")
#             extracted_text = ""
            
#             for block in text_dict["blocks"]:
#                 if block.get("type") == 0:  # Text block
#                     for line in block.get("lines", []):
#                         line_text = ""
#                         for span in line.get("spans", []):
#                             span_text = span.get("text", "")
#                             if span_text.strip():
#                                 line_text += span_text
#                         if line_text.strip():
#                             extracted_text += line_text + "\n"
            
#             # Fallback to simple extraction if structured fails
#             if not extracted_text.strip():
#                 extracted_text = page.get_text()
            
#             return self.clean_text(extracted_text)
#         except Exception as e:
#             print(f"Error extracting page {page_num + 1}: {e}")
#             return ""

#     def chunk_page_text(self, text: str) -> List[str]:
#         """FIX 6: Smart chunking for Bengali text."""
#         if len(text) <= 1000:
#             return [text] if text.strip() else []
        
#         # Split on Bengali sentence endings
#         sentences = re.split(r'[।!?]', text)
#         chunks = []
#         current_chunk = ""
        
#         for sentence in sentences:
#             sentence = sentence.strip()
#             if not sentence:
#                 continue
                
#             # If adding this sentence would make chunk too long
#             if len(current_chunk) + len(sentence) > 800 and current_chunk:
#                 chunks.append(current_chunk.strip())
#                 current_chunk = sentence
#             else:
#                 current_chunk += " " + sentence if current_chunk else sentence
        
#         # Add the last chunk
#         if current_chunk.strip():
#             chunks.append(current_chunk.strip())
        
#         # Filter out too-short chunks
#         return [chunk for chunk in chunks if len(chunk.strip()) > 100]

#     def embed_and_store_pdf(self, file_path: str, doc_name: str):
#         try:
#             doc = fitz.open(file_path)
#             print(f"Processing PDF {file_path} with {len(doc)} pages")

#             vectors_batch = []
#             batch_size = 30  # Smaller batch size for stability
#             total_stored = 0

#             for page_num in range(len(doc)):
#                 page_text = self.extract_clean_text_from_pdf_page(doc, page_num)

#                 if not page_text or len(page_text) < 100:
#                     print(f"Skipping page {page_num + 1}: insufficient content")
#                     continue

#                 # FIX 7: Chunk each page for better retrieval
#                 chunks = self.chunk_page_text(page_text)
#                 print(f"Page {page_num + 1}: Created {len(chunks)} chunks")

#                 for chunk_idx, chunk in enumerate(chunks):
#                     embedding = self.create_embedding(chunk)
#                     if not embedding:
#                         continue

#                     sanitized_name = self.sanitize_vector_id(doc_name)
#                     unique_id = self.generate_unique_id()
#                     vector_id = f"{sanitized_name}_p{page_num}_c{chunk_idx}_{unique_id}"

#                     metadata = {
#                         "file_name": os.path.basename(file_path),
#                         "document_name": doc_name,
#                         "page_number": page_num + 1,
#                         "chunk_index": chunk_idx,
#                         "text": chunk[:8000],  # Store full chunk
#                         "char_count": len(chunk),
#                         "file_type": "pdf",
#                         # FIX 8: Add language detection
#                         "language": "bengali" if any('\u0980' <= c <= '\u09FF' for c in chunk) else "mixed"
#                     }

#                     vectors_batch.append({
#                         "id": vector_id,
#                         "values": embedding,
#                         "metadata": metadata
#                     })
#                     total_stored += 1

#                     if len(vectors_batch) >= batch_size:
#                         self.index.upsert(vectors=vectors_batch)
#                         print(f"Uploaded batch of {len(vectors_batch)} vectors")
#                         vectors_batch = []

#             # Upload remaining vectors
#             if vectors_batch:
#                 self.index.upsert(vectors=vectors_batch)
#                 print(f"Uploaded final batch of {len(vectors_batch)} vectors")

#             doc.close()
#             print(f"✅ Finished processing: {doc_name} - Total chunks stored: {total_stored}")

#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")

# # === MAIN EXECUTION ===
# if __name__ == "__main__":
#     pdf_path = "C:\\Users\\hasan\\Downloads\\10ms Assesment\\HSC26-Bangla1st-Paper.pdf"
    
#     # FIX 9: Use different index name to avoid conflicts
#     vectorizer = WordVectorizerOpenAIPinecone(
#         pinecone_index_name="hsc26-bangla-fixed"
#     )
#     vectorizer.embed_and_store_pdf(file_path=pdf_path, doc_name="HSC26 Bangla 1st Paper")





import random
import string
import fitz  # PyMuPDF
from typing import List
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import re

load_dotenv()

class WordVectorizerOpenAIPinecone:
    def __init__(self, pinecone_index_name: str):
        self.index_name = pinecone_index_name
        # Use newer embedding model that works better with Bengali
        self.embedding_model = "text-embedding-3-small"
        
        # Set OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client based on version
        self._init_openai_client()

        self.pc = Pinecone(os.environ.get("PINECONE_API_KEY"))
        if pinecone_index_name not in [i.name for i in self.pc.list_indexes()]:
            print(f"Creating new index: {pinecone_index_name}")
            self.pc.create_index(
                name=pinecone_index_name,
                dimension=1536,  # text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(pinecone_index_name)

    def _init_openai_client(self):
        """Initialize OpenAI client compatible with different versions."""
        try:
            # Try new client (v1.0+)
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.use_new_client = True
            print("Using new OpenAI client (v1.0+)")
        except ImportError:
            # Fall back to old client (v0.x)
            self.client = None
            self.use_new_client = False
            print("Using legacy OpenAI client (v0.x)")

    def generate_unique_id(self) -> str:
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    def sanitize_vector_id(self, vector_id: str) -> str:
        # Better sanitization that preserves Bengali characters
        sanitized = re.sub(r'[^\w\d\u0980-\u09FF]', '_', vector_id)
        sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
        if len(sanitized) > 64:
            prefix = sanitized[:50]
            suffix = self.generate_unique_id()
            sanitized = f"{prefix}_{suffix}"
        return sanitized

    def clean_text(self, text: str) -> str:
        """Improved cleaning for Bengali text."""
        if not text or len(text.strip()) < 20:
            return ""
            
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        # Remove page numbers in both Bengali and English
        text = re.sub(r'(পৃষ্ঠা\s*\d+|Page\s*\d+|পাতা\s*\d+)', '', text, flags=re.IGNORECASE)
        
        # Remove PDF artifacts
        text = re.sub(r'[\x0c\x00-\x08\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Clean up excessive punctuation
        text = re.sub(r'[।]{2,}', '।', text)
        
        return text.strip()

    def create_embedding(self, text: str) -> List[float]:
        """Create embeddings compatible with both old and new OpenAI clients."""
        try:
            if not text or len(text.strip()) < 10:
                return []
            
            # Better chunking strategy for long text
            max_chunk_size = 8000  # Conservative limit
            
            if len(text) <= max_chunk_size:
                # Single chunk
                if self.use_new_client:
                    # New client (v1.0+)
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=text,
                        encoding_format="float"
                    )
                    return response.data[0].embedding
                else:
                    # Legacy client (v0.x)
                    response = openai.Embedding.create(
                        model=self.embedding_model,
                        input=text
                    )
                    return response['data'][0]['embedding']
            else:
                # Multiple chunks with overlap
                chunks = []
                overlap = 200
                for i in range(0, len(text), max_chunk_size - overlap):
                    chunk = text[i:i + max_chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    if i + max_chunk_size >= len(text):
                        break

                if not chunks:
                    return []

                embeddings = []
                for chunk in chunks:
                    if self.use_new_client:
                        # New client (v1.0+)
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=chunk,
                            encoding_format="float"
                        )
                        embeddings.append(response.data[0].embedding)
                    else:
                        # Legacy client (v0.x)
                        response = openai.Embedding.create(
                            model=self.embedding_model,
                            input=chunk
                        )
                        embeddings.append(response['data'][0]['embedding'])

                # Average the embeddings
                if embeddings:
                    averaged_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
                    return averaged_embedding
                else:
                    return []

        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []

    def extract_clean_text_from_pdf_page(self, doc: fitz.Document, page_num: int) -> str:
        """Better text extraction for Bengali PDFs."""
        try:
            page = doc.load_page(page_num)
            
            # Try structured extraction first (better for Bengali)
            text_dict = page.get_text("dict")
            extracted_text = ""
            
            for block in text_dict["blocks"]:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            if span_text.strip():
                                line_text += span_text
                        if line_text.strip():
                            extracted_text += line_text + "\n"
            
            # Fallback to simple extraction if structured fails
            if not extracted_text.strip():
                extracted_text = page.get_text()
            
            return self.clean_text(extracted_text)
        except Exception as e:
            print(f"Error extracting page {page_num + 1}: {e}")
            return ""

    def chunk_page_text(self, text: str) -> List[str]:
        """Smart chunking for Bengali text."""
        if len(text) <= 1000:
            return [text] if text.strip() else []
        
        # Split on Bengali sentence endings
        sentences = re.split(r'[।!?]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would make chunk too long
            if len(current_chunk) + len(sentence) > 800 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out too-short chunks
        return [chunk for chunk in chunks if len(chunk.strip()) > 100]

    def embed_and_store_pdf(self, file_path: str, doc_name: str):
        try:
            doc = fitz.open(file_path)
            print(f"Processing PDF {file_path} with {len(doc)} pages")

            vectors_batch = []
            batch_size = 30  # Smaller batch size for stability
            total_stored = 0

            for page_num in range(len(doc)):
                page_text = self.extract_clean_text_from_pdf_page(doc, page_num)

                if not page_text or len(page_text) < 100:
                    print(f"Skipping page {page_num + 1}: insufficient content")
                    continue

                # Chunk each page for better retrieval
                chunks = self.chunk_page_text(page_text)
                print(f"Page {page_num + 1}: Created {len(chunks)} chunks")

                for chunk_idx, chunk in enumerate(chunks):
                    embedding = self.create_embedding(chunk)
                    if not embedding:
                        continue

                    sanitized_name = self.sanitize_vector_id(doc_name)
                    unique_id = self.generate_unique_id()
                    vector_id = f"{sanitized_name}_p{page_num}_c{chunk_idx}_{unique_id}"

                    metadata = {
                        "file_name": os.path.basename(file_path),
                        "document_name": doc_name,
                        "page_number": page_num + 1,
                        "chunk_index": chunk_idx,
                        "text": chunk[:8000],  # Store full chunk
                        "char_count": len(chunk),
                        "file_type": "pdf",
                        # Add language detection
                        "language": "bengali" if any('\u0980' <= c <= '\u09FF' for c in chunk) else "mixed"
                    }

                    vectors_batch.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                    total_stored += 1

                    if len(vectors_batch) >= batch_size:
                        self.index.upsert(vectors=vectors_batch)
                        print(f"Uploaded batch of {len(vectors_batch)} vectors")
                        vectors_batch = []

            # Upload remaining vectors
            if vectors_batch:
                self.index.upsert(vectors=vectors_batch)
                print(f"Uploaded final batch of {len(vectors_batch)} vectors")

            doc.close()
            print(f"✅ Finished processing: {doc_name} - Total chunks stored: {total_stored}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    pdf_path = "C:\\Users\\hasan\\Downloads\\10ms Assesment\\HSC26-Bangla1st-Paper.pdf"
    
    # Use different index name to avoid conflicts
    vectorizer = WordVectorizerOpenAIPinecone(
        pinecone_index_name="hsc26-bangla-fixed"
    )
    vectorizer.embed_and_store_pdf(file_path=pdf_path, doc_name="HSC26 Bangla 1st Paper")