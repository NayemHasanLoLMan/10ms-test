import os
import re
from collections import deque
from typing import List, Tuple
import openai
from dotenv import load_dotenv
from langdetect import detect
from pinecone import Pinecone

# === Load API Keys ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# === Initialize OpenAI Client ===
def init_openai_client():
    """Initialize OpenAI client compatible with different versions."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        use_new_client = True
        print("Using new OpenAI client (v1.0+)")
    except ImportError:
        client = None
        use_new_client = False
        print("Using legacy OpenAI client (v0.x)")
    return client, use_new_client

client, use_new_client = init_openai_client()

# === Initialize Pinecone Index ===
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("hsc26-bangla-fixed")

# === Model Settings ===
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# === Conversation Memory ===
class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)

    def add(self, user_msg, bot_msg):
        self.history.append({"user": user_msg, "bot": bot_msg})

    def get_context(self) -> str:
        if not self.history:
            return ""
        return "\n".join([f"ব্যবহারকারী: {m['user']}\nAI: {m['bot']}" for m in self.history])

memory = ConversationMemory()

# === Embed Text ===
def embed_text(text: str) -> List[float]:
    """Create embeddings compatible with both old and new OpenAI clients."""
    try:
        if use_new_client:
            # New client (v1.0+)
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=text,
                encoding_format="float"
            )
            return resp.data[0].embedding
        else:
            # Legacy client (v0.x)
            resp = openai.Embedding.create(
                model=EMBED_MODEL,
                input=text
            )
            return resp['data'][0]['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

# === Retrieve Similar Chunks ===
def retrieve_chunks(query: str, top_k=15, score_threshold=0.25) -> List[Tuple[str, float]]:
    """Retrieve relevant chunks from Pinecone."""
    vec = embed_text(query)
    if not vec:
        return []
    
    try:
        resp = index.query(vector=vec, top_k=top_k, include_metadata=True, include_values=False)
        chunks = [
            (m["metadata"].get("text", ""), m["score"])
            for m in resp.get("matches", [])
            if m["score"] >= score_threshold and m["metadata"].get("text")
        ]
        print(f"Retrieved {len(chunks)} chunks with scores: {[round(s, 3) for _, s in chunks[:5]]}")
        return chunks
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

# === Rerank with GPT ===
def rerank_chunks(chunks: List[str], query: str) -> List[str]:
    """Rerank chunks using GPT."""
    if not chunks:
        return []
    
    is_bengali = any('\u0980' <= c <= '\u09FF' for c in query)
    
    if is_bengali:
        prompt = f"নিচের অনুচ্ছেদগুলি প্রশ্নের সাথে কতটা প্রাসঙ্গিক তার ভিত্তিতে সাজান। সবচেয়ে প্রাসঙ্গিকটি আগে রাখুন।\n\nপ্রশ্ন: {query}\n\n"
    else:
        prompt = f"Rank these passages by relevance to the question. Most relevant first.\n\nQuestion: {query}\n\n"
    
    for i, chunk in enumerate(chunks[:10], 1):
        prompt += f"[{i}] {chunk[:500]}...\n\n"
    
    try:
        if use_new_client:
            # New client (v1.0+)
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = resp.choices[0].message.content
        else:
            # Legacy client (v0.x)
            resp = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = resp['choices'][0]['message']['content']
        
        ids = [int(m) - 1 for m in re.findall(r"\[(\d+)\]", text)]
        reranked = [chunks[i] for i in ids if 0 <= i < len(chunks)]
        
        for i, chunk in enumerate(chunks):
            if i not in ids and chunk not in reranked:
                reranked.append(chunk)
                
        return reranked[:5]
    except Exception as e:
        print(f"Reranking error: {e}")
        return chunks[:5]

# === Build Context ===
def build_context(chunks: List[str], max_chars=5000) -> str:
    """Build context with deduplication."""
    seen, ctx = set(), ""
    for c in chunks:
        if not c.strip():
            continue
            
        norm = re.sub(r"\s+", " ", c.strip().lower())[:100]
        if norm in seen:
            continue
        
        if len(ctx) + len(c) > max_chars:
            break
        
        seen.add(norm)
        ctx += c + "\n\n"
    
    return ctx.strip()

# === Generate Final Answer ===
def generate_answer(context: str, query: str, history: str) -> str:
    """Generate answer using GPT."""
    is_bengali = any('\u0980' <= c <= '\u09FF' for c in query)
    
    if is_bengali:
        system_prompt = """আপনি একটি বাংলা ভাষার সহায়ক AI। আপনার কাজ হল প্রদত্ত তথ্যের ভিত্তিতে প্রশ্নের সঠিক উত্তর দেওয়া।

নিয়মাবলী:
- শুধুমাত্র দেওয়া তথ্য ব্যবহার করুন
- সংক্ষিপ্ত ও সরাসরি উত্তর দিন
- তথ্য না থাকলে "আমি নিশ্চিত নই" বলুন
- অতিরিক্ত ব্যাখ্যা দেবেন না"""
        user_prompt = f"""তথ্য:
{context}

পূর্ববর্তী কথোপকথন:
{history}

প্রশ্ন: {query}

উত্তর (শুধুমাত্র প্রয়োজনীয় তথ্য):"""
    else:
        system_prompt = """You are a helpful AI assistant. Answer questions based only on the provided context.

Rules:
- Use only the given information
- Give brief and direct answers  
- Say "I'm not sure" if information is not available
- Don't add extra explanations"""
        user_prompt = f"""Context:
{context}

Previous conversation:
{history}

Question: {query}

Answer (essential information only):"""

    try:
        if use_new_client:
            # New client (v1.0+)
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        else:
            # Legacy client (v0.x)
            resp = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return "দুঃখিত, আমি নিশ্চিত নই।" if is_bengali else "I'm not sure."

# === Main Chat Function ===
def chat(user_input: str) -> str:
    """Process user query and generate response."""
    print(f"\n🔍 Processing query: {user_input}")
    
    history = memory.get_context()
    chunks = retrieve_chunks(user_input, top_k=20, score_threshold=0.2)
    
    if not chunks:
        print("❌ No relevant chunks found")
        reply = "আমি এই প্রশ্নের উত্তর খুঁজে পাচ্ছি না। আরো স্পষ্টভাবে প্রশ্ন করুন।"
    else:
        print(f"✅ Found {len(chunks)} relevant chunks")
        texts = [t for t, _ in chunks]
        reranked = rerank_chunks(texts, user_input)
        ctx = build_context(reranked)
        print(f"📄 Context length: {len(ctx)} characters")
        reply = generate_answer(ctx, user_input, history)

    memory.add(user_input, reply)
    return reply

# === Command-line Interface ===
def interactive():
    print("🔍 Ask your question (Bangla or English). Type 'exit' to quit.")
    print("প্রশ্ন করুন (বাংলা বা ইংরেজি)। 'exit' লিখে বের হন।\n")
    
    while True:
        u = input("You/আপনি: ").strip()
        if u.lower() == "exit":
            break
        if not u:
            continue
            
        ans = chat(u)
        print(f"Bot: {ans}\n")

if __name__ == "__main__":
    print("🧪 Testing system with sample query...")
    test_result = chat("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?")
    print(f"Test result: {test_result}")
    print("\n" + "="*50 + "\n")
    
    interactive()