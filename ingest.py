import weaviate
from llama_index import StorageContext, SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import box
import yaml

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_documents(docs_path):
    reader = SimpleDirectoryReader(input_dir=docs_path)

    docs = reader.load_data()
    print(f"Loaded {len(docs)} docs")

    return docs


def load_embedding_model(model_name):
    embed = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name=model_name)
    )
    return embed


def build_index(weaviate_client, embed_model, docs, index_name):
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs_index = VectorStoreIndex.from_documents(
        docs,
        service_context=service_context,
        storage_context=storage_context,
    )

    return docs_index


if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf8') as configuration:
        cfg = box.Box(yaml.safe_load(configuration))

    print("Connecting to Weaviate")
    client = weaviate.Client(cfg.WEAVIATE_URL)

    print("Loading documents...")
    documents = load_documents(cfg.DATA_PATH)

    print("Loading embedding model...")
    embeddings = load_embedding_model(model_name=cfg.EMBEDDINGS)

    print("Building index...")
    index = build_index(client, embeddings, documents, cfg.INDEX_NAME)

