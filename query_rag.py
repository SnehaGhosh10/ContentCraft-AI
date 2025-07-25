import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env file
load_dotenv()

# Get API key
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file. Please set it correctly.")

# Load FAISS index
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

def generate_with_rag(user_query: str) -> str:
    # Retrieve relevant documents
    docs = db.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt for LLaMA-3 (Groq)
    prompt = f"""
You are a highly experienced content strategist writing for the Indian digital finance audience.

TASK:
Write a blog post or social media content based on the user query.

GOAL:
- Educate the audience
- Maintain a friendly but informative tone
- Focus on Indian context and relevance

CONTEXT:
{context}

INSTRUCTION:
Using the context above, generate a 200–300 word article or post for the topic: "{user_query}"

REQUIREMENTS:
- Use clear, simple language
- Include bullet points if helpful
- Avoid hallucinations; rely only on provided context
"""

    # Groq API call
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 700
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, headers=headers)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"❌ API Error {response.status_code}: {response.text}"
