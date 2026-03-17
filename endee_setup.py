from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    with open("data/medical_data.txt","r") as f:
        texts = f.readlines()

    embeddings = model.encode(texts)
    return texts, embeddings

texts, embeddings = load_data()

def search(query):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    best_index = np.argmax(scores)
    return texts[best_index]