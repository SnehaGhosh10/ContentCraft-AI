import streamlit as st
from query_rag import generate_with_rag


# Page config
st.set_page_config(page_title="AI Content Generator", layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #dbeafe, #fdf2f8);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border: 2px solid #3b82f6;
        border-radius: 0.5rem;
        font-size: 1rem;
        padding: 0.6rem;
        background-color: #f0f9ff;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .stMarkdown h1 {
        color: #1e3a8a;
    }
    .generated-box {
        background-color: #f9fafb;
        border-left: 6px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Segoe UI', sans-serif;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Title and instructions
st.markdown("<h1 style='text-align: center;'>🧠 AI Content Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Generate blog or social media content using <b>LLaMA-3 + FAISS + Groq</b></p>", unsafe_allow_html=True)
st.write("")

# Input
topic = st.text_input("📌 Enter your topic", placeholder="e.g., Benefits of SIP Investing")

# Button
if st.button("Generate Content") and topic:
    with st.spinner("🔮 Generating using LLaMA-3..."):
        output = generate_with_rag(topic)
        st.success("✅ Content Generated")

        # Output display
        st.markdown("### 📝 Generated Content")
        st.markdown(f"<div class='generated-box'>{output}</div>", unsafe_allow_html=True)
