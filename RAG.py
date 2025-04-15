import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.neighbors import NearestNeighbors
import numpy as np

# --- Load models ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return embedder, tokenizer, model

embedder, tokenizer, generator = load_models()

# --- Sample documents ---
documents = [
    "The Eiffel Tower is located in Paris.",
    "Python is a programming language.",
    "Transformers are state-of-the-art NLP models.",
    "The moon orbits the Earth.",
    "The capital of France is Paris.",
]

# --- Build index using sklearn ---
@st.cache_resource
def build_index(docs):
    embeddings = embedder.encode(docs, convert_to_numpy=True)
    index = NearestNeighbors(n_neighbors=3, metric='cosine').fit(embeddings)
    return index, embeddings

index, doc_embeddings = build_index(documents)

# --- Retrieval function ---
def retrieve(query, k=2):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.kneighbors(query_embedding, n_neighbors=k)
    return [documents[i] for i in indices[0]]

# --- Generator function ---
def generate_answer(query, context_docs):
    context = " ".join(context_docs)
    prompt = f"question: {query} context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = generator.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit App UI ---
st.title("🧠 RAG Chatbot")
st.write("Ask a question and get an answer from the mini knowledge base.")

query = st.text_input("🔎 Your Question:")

if query:
    with st.spinner("Searching and generating answer..."):
        retrieved = retrieve(query)
        answer = generate_answer(query, retrieved)

    st.subheader("📄 Retrieved Documents")
    for doc in retrieved:
        st.markdown(f"- {doc}")

    st.subheader("🧠 Answer")
    st.success(answer)
