from langchain_chroma import Chroma
from langchain.schema import Document 
from typing import List
import chromadb

class VectorStore:
    def __init__(self,embeddings,persist_directory: str = "chroma_db"):
        self._embeddings = embeddings
        self.retriever = None
        self.vectorstore = None
        self.persist_directory = persist_directory

    def load_retriever(self,k:int=3):
        client = chromadb.PersistentClient(path=self.persist_directory)
        

        if client is not None:
            # You can get a collection (or create if needed)
            collection_name = self.persist_directory
            try:
                vector_store_from_client = Chroma(
                        client=client,
                        collection_name=collection_name,
                        embedding_function=self._embeddings,
                    )
            except ValueError:
                raise RuntimeError(f"Collection '{collection_name}' not found. Add documents first.")

            # Wrap as retriever (for LangChain or custom retriever)
            self.retriever = vector_store_from_client.as_retriever(search_kwargs={"k": k})
            return self.retriever
        else:
            raise RuntimeError("No Vector Database found. Please add documents first.")
    
    def add_documents(self,docs:List[Document]):
        print(docs[:2])
        self.vectorstore = Chroma.from_documents(
                docs,
                embedding=self._embeddings,
                persist_directory=self.persist_directory ,  # local folder,
                collection_name=self.persist_directory
            )
        # self.vectorstore.persist()
        # return self.vectorestore
