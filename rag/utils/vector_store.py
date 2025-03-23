### Build Index
import weaviate
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore


from rag.utils.embedding import get_embeddings

class LocalVectorStore:
    client = weaviate.connect_to_local()
    vector_store = None

    def ingest_fs(self, path: str):
        loader = DirectoryLoader(path=path, glob="**/[!.]*", show_progress=True)

        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")


        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        #for blob in documents:
        #    print(blob)

        self.vector_store = WeaviateVectorStore.from_documents(
            docs,
            get_embeddings(),
            client=self.client,
            index_name="FolderDocs"
        )

        print("Documents have been embedded and stored in Weaviate.")


    def load_index(self, index_name):
        self.vector_store = WeaviateVectorStore(client=self.client, index_name=index_name, text_key="text", embedding=get_embeddings())


    def close(self):
        self.client.close()

    def retriever(self):
        return self.vector_store.as_retriever()