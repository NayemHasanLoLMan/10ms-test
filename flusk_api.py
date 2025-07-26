from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("bangla-docx-embeddings")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Initialize conversation history with a maximum length of 5
conversation_history = []
MAX_HISTORY_LENGTH = 5
HISTORY_FILE = 'conversation_history.json'


# Load conversation history from the JSON file at server start
def load_conversation_history():
    global conversation_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                conversation_history = json.load(f)
            except json.JSONDecodeError:
                print("Error loading conversation history from file, starting with an empty history.")

# Save conversation history to the JSON file
def save_conversation_history():
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=4)



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

    conversation_prompt = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in conversation_history])

    prompt = f"""You are an expert assistant helping to answer questions based on understanding document content and what it implies. Only give the answer don't provide any sources or extra context. 

    
    CONVERSATION HISTORY (use this to understand the previous context and what the user is asking):
    {conversation_prompt}

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
    7. Answer in the same language as the QUESTION: {query} (if needed translate the answer to {query} language)
    8. Be specific in your response

    ANSWER:"""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text, sources_info
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, having some issue with generating answare.", []

def get_answer(query):
    """Fetches the answer to a query based on the document chunks in the knowledge base."""
    print(f"question: {query}")

    relevant_chunks = search_knowledge_base(query, top_k=30)

    if not relevant_chunks:
        return "No, releted answare found the given question", []

    context_groups = group_chunks_by_context(relevant_chunks, max_context_length=5000)
    answer, sources = answer_question_with_context(query, context_groups)


    # Append to conversation history
    conversation_history.append({"question": query, "answer": answer})
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history.pop(0)


    # Save updated conversation history to JSON
    save_conversation_history()

    return answer, sources


@app.route('/ask', methods=['POST'])
def ask_question():
    """Main endpoint to ask questions"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400

        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Missing required field: query'
            }), 400

        query = data['query'].strip()
        
        if not query:
            return jsonify({
                'error': 'Query cannot be empty'
            }), 400

        # Get answer using existing function
        logger.info(f"Processing query: {query}")
        answer, sources = get_answer(query)

        # Format sources for response
        formatted_sources = []
        seen_blocks = set()
        
        for block, score in sources:
            if block not in seen_blocks:
                formatted_sources.append({
                    'block': block,
                    'score': round(score, 4)
                })
                seen_blocks.add(block)

        response_data = {
            'query': query,
            'answer': answer,
        }

        logger.info(f"Successfully processed query with {len(formatted_sources)} sources")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405

if __name__ == '__main__':
    # Check environment variables
    required_env_vars = ['PINECONE_API_KEY', 'GEMINI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)

    load_conversation_history()
    
    logger.info("Starting Bangla Q&A API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)