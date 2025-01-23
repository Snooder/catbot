from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a text embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample proprietary data
docs = [
    "Our business hours are from 9 AM to 5 PM.",
    "We provide logistics services for e-commerce.",
    "Customer support can be reached at support@example.com."
]

# Convert text to vector embeddings
doc_vectors = np.array([model.encode(doc) for doc in docs])

# Store in FAISS
index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(doc_vectors)

# Search query
query = "What time does the company open?"
query_vector = np.array([model.encode(query)])
D, I = index.search(query_vector, k=1)

# Return the most relevant document
print("Best Match:", docs[I[0][0]])
