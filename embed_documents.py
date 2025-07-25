import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your text file containing domain knowledge
loader = TextLoader("articles.txt")
documents = loader.load()

# Split the documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# Create embeddings using HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build the FAISS index
db = FAISS.from_documents(split_docs, embedding_model)

# Save the index to disk
db.save_local("faiss_index")
print("✅ FAISS index created and saved locally.")

