from langchain_ollama import OllamaEmbeddings



embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def get_embeddings():
    return embeddings