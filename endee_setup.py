from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    with open("medical_data.txt", "r") as f:
        texts = f.readlines()

    embeddings = model.encode(texts)
    return texts, embeddings

texts, embeddings = load_data()

def search(query):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    index = similarities.argmax()
    return texts[index]
