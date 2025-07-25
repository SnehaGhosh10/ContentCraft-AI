from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load plain text from file
loader = TextLoader("articles.txt") 
docs = loader.load()

# Split documents into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index from embedded docs
db = FAISS.from_documents(split_docs, embedding)

# Save index to local folder
os.makedirs("faiss_index", exist_ok=True)
db.save_local("faiss_index")

print(f"✅ FAISS index created with {len(split_docs)} chunks.")
