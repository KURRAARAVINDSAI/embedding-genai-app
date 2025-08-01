import streamlit as st
import os

from chunk_utils import character_split, recursive_split
from embedding_utils import get_embeddings

def load_css():
    try:
        with open("frontend/style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found.")

def load_js():
    try:
        with open("frontend/script.js", "r") as f:
            st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("script.js not found.")

def load_html():
    try:
        with open("frontend/index(3).html", "r") as f:
            html = f.read()
            st.components.v1.html(html, height=150, scrolling=False)
    except FileNotFoundError:
        st.warning("index.html not found.")

# Streamlit app configuration
st.set_page_config(page_title="Text Chunking & Embedding Visualizer", layout="wide")

# Load frontend assets
load_css()
load_js()
load_html()

# Title and options
st.title("📌 Text Chunking & Embedding Visualizer")
st.markdown("### Choose a chunking method")

chunking_method = st.radio("Select method:", ["Character Split", "Recursive Split"], horizontal=True)
chunk_size = st.slider("Chunk size", min_value=50, max_value=300, value=100, step=10)
chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=100, value=20, step=10)

# Load dataset
if os.path.exists("dataset.txt"):
    with open("dataset.txt", "r", encoding="utf-8") as file:
        text = file.read()
else:
    st.error("dataset.txt file not found in the root directory.")
    st.stop()

# Chunking
if chunking_method == "Character Split":
    chunks = character_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
elif chunking_method == "Recursive Split":
    chunks = recursive_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
else:
    chunks = []

# Display chunked text
st.markdown("### 🔹 Chunked Text")
if chunks:
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}:** {chunk}")
else:
    st.info("No chunks to display.")

# Embedding generation
# Generate embeddings and show vectors
if st.button("Generate Embeddings"):
    embeddings = get_embeddings(chunks)
    st.success("✅ Embeddings generated!")

    st.markdown("### 🔸 Embedding Vectors (first 2 chunks)")
    for i, emb in enumerate(embeddings[:2]):
        st.markdown(f"**Chunk {i+1} Vector:**")
        # Split 384-d vector into ranges [0-100], [100-200], etc.
        ranges = [(0, 100), (100, 200), (200, 300), (300, 384)]
        for start, end in ranges:
            vector_slice = emb[start:end]
            st.code(f"[{start} - {end}]: {vector_slice}", language="python")