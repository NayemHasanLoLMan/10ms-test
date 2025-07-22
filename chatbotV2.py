import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langdetect import detect
from typing import List
from collections import deque

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
serverless_spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "hsc26-bangla-1st-paper"
index = pc.Index(index_name)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Conversation memory class
class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)

    def add_interaction(self, user_message, bot_response):
        self.history.append({"user": user_message, "bot": bot_response})

    def get_conversation_context(self):
        context = ""
        for i in self.history:
            context += f"User: {i['user']}\nBot: {i['bot']}\n\n"
        return context

memory = ConversationMemory(max_history=5)

# Generate embedding with Gemini
def embed_text_with_gemini(text: str) -> List[float]:
    if text.strip():
        response = genai.embed_content(
            model="text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']
    return None

# Retrieve relevant document chunks
def retrieve_relevant_documents(query: str, top_k: int = 15, score_threshold: float = 0.75) -> List[str]:
    query_embedding = embed_text_with_gemini(query)
    if not query_embedding:
        return []

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False  # Optional: skips returning embedding vectors
    )

    # Filter by similarity score threshold (> 0.75)
    filtered_texts = [
        match['metadata'].get("text", "")
        for match in results['matches']
        if match['score'] > score_threshold
    ]

    return filtered_texts

# Deduplicate document passages
def deduplicate_passages(passages: List[str]) -> List[str]:
    seen = set()
    unique = []
    for p in passages:
        norm = re.sub(r'\s+', ' ', p.lower().strip())
        if norm not in seen:
            seen.add(norm)
            unique.append(p)
    return unique

# Build context for the model
def build_context(passages: List[str], max_length=6000) -> str:
    unique_passages = deduplicate_passages(passages)
    context = " ".join(unique_passages)
    return context[:max_length]

# Generate a Gemini response based on query language
def generate_response(context: str, query: str, conversation_history: str, lang: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        if lang == "bn":
            prompt = f"""
        আপনার কাজ হলো শুধুমাত্র নিচে প্রদত্ত প্রাসঙ্গিক তথ্যের উপর ভিত্তি করে প্রশ্নের উত্তর দেয়া। 
        তথ্যে যদি সরাসরি উত্তর না থাকে, তাহলে বলুন: "আমি নিশ্চিত না।" 
        অনুগ্রহ করে কোনো অতিরিক্ত ব্যাখ্যা বা তথ্য যুক্ত করবেন না।

        [আগের কথোপকথন]
        {conversation_history}

        [প্রাসঙ্গিক তথ্য]
        {context}

        [প্রশ্ন]
        {query}

        [উত্তর]
        """
        else:
            prompt = f"""
        Your task is to answer the question **only** using the relevant information provided below.
        If the answer is not directly available in the context, respond with: "I am not sure."
        Do not add any explanation, speculation, or external knowledge.

        [Conversation history]
        {conversation_history}

        [Relevant information]
        {context}

        [Question]
        {query}

        [Answer]
        """
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=300
            )
        )
        return response.text.strip()

    except Exception as e:
        print(f"Error: {e}")
        return "দুঃখিত, আমি এই প্রশ্নের উত্তর দিতে পারছি না।"


# Chat function
def chat_with_rag_system(user_message: str) -> str:
    try:
        lang = detect(user_message)
    except:
        lang = "bn"  # default fallback

    conversation_context = memory.get_conversation_context()
    passages = retrieve_relevant_documents(user_message)
    context = build_context(passages)

    response = generate_response(context, user_message, conversation_context, lang)
    memory.add_interaction(user_message, response)
    return response

# Interactive CLI
def interactive_chat():

    user_input = input("User: ")
    response = chat_with_rag_system(user_input)
    print(f"Chatbot: {response}\n")

# Entry point
if __name__ == "__main__":
    interactive_chat()
