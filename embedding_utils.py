from sentence_transformers import SentenceTransformer
import torch

def get_embeddings(chunks):
    # Safely detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Proper way to load model without hitting meta tensor error
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model = model.to(device)

    # Encode text chunks into embeddings
    embeddings = model.encode(chunks, convert_to_tensor=True)

    return embeddings