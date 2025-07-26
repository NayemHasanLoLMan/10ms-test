import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("bangla-docx-embeddings")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def embed_query(query):
    """Create embedding for the search query"""
    try:
        response = genai.embed_content(
            model='gemini-embedding-001',
            content=query,
            task_type="retrieval_query"
        )
        return response['embedding']
    except Exception as e:
        print(f"Error creating query embedding: {e}")
        return None

def search_knowledge_base(query, top_k=30):
    """Search for relevant chunks in the knowledge base"""
    query_embedding = embed_query(query)

    if not query_embedding:
        return []

    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        relevant_chunks = []
        for match in results.matches:
            if match.score > 0.5:
                relevant_chunks.append({
                    'text': match.metadata.get('text', ''),
                    'block': match.metadata.get('block', 'Unknown'),
                    'source': match.metadata.get('source', 'Unknown'),
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'score': match.score
                })

        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        return relevant_chunks

    except Exception as e:
        print(f"Search error: {e}")
        return []

def group_chunks_by_context(chunks, max_context_length=5000):
    if not chunks:
        return []

    block_groups = {}
    for chunk in chunks:
        block = chunk['block']
        if block not in block_groups:
            block_groups[block] = []
        block_groups[block].append(chunk)

    for block in block_groups:
        block_groups[block].sort(key=lambda x: x['chunk_index'])

    context_groups = []
    current_group = []
    current_length = 0

    sorted_blocks = sorted(block_groups.keys(),
                           key=lambda b: max(chunk['score'] for chunk in block_groups[b]),
                           reverse=True)

    for block in sorted_blocks:
        block_chunks = block_groups[block]
        block_text = " ".join([chunk['text'] for chunk in block_chunks])

        if current_length + len(block_text) <= max_context_length:
            current_group.extend(block_chunks)
            current_length += len(block_text)
        else:
            if current_group:
                context_groups.append(current_group)
            current_group = block_chunks[:]
            current_length = len(block_text)

    if current_group:
        context_groups.append(current_group)

    return context_groups

def answer_question_with_context(query, context_groups):
    if not context_groups:
        return "Not enough relevent chunk of information found", []

    all_contexts = []
    sources_info = []

    for i, group in enumerate(context_groups):
        group_text = ""
        group_blocks = set()

        for chunk in group:
            group_text += chunk['text'] + " "
            group_blocks.add(chunk['block'])

        all_contexts.append(f"Context {i+1} (Blocks: {', '.join(map(str, sorted(group_blocks)))}):\n{group_text.strip()}")
        sources_info.extend([(chunk['block'], chunk['score']) for chunk in group])

    full_context = "\n\n".join(all_contexts)

    prompt = f"""You are an expert assistant helping to answer questions based on understanding document content and what it implies. Only give the answer don't provide any sources or extra context. 

    CONTEXTS FROM DOCUMENT:
    {full_context}

    QUESTION: {query}

    INSTRUCTIONS:
    1. Analyze all the provided contexts carefully
    2. Use information from multiple contexts when relevant
    3. Provide a comprehensive answer based only on the given contexts
    4. If contexts contain conflicting information, mention it
    5. Understand the context and what it implies and provide the best possible answer
    6. If the contexts don't contain enough information, say so clearly
    7. Answer in the same language as the question
    8. Be specific in your response

    ANSWER:"""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text, sources_info
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, having some issue with generating answare", []

def get_answer(query):
    """Fetches the answer to a query based on the document chunks in the knowledge base."""
    print(f" question : {query}")

    relevant_chunks = search_knowledge_base(query, top_k=30)

    if not relevant_chunks:
        return "No, releted answare found the given question", []

    context_groups = group_chunks_by_context(relevant_chunks, max_context_length=5000)
    answer, sources = answer_question_with_context(query, context_groups)

    return answer, sources

# Example Usage:
if __name__ == "__main__":
    query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    answer, sources = get_answer(query)
    print("Answer:", answer)
