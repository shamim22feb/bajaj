from langchain_community.document_loaders import PyMuPDFLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class PDFParser():
    def __init__(self,filepath):
        self.filepath =filepath

    def process_file(self):
        loader = PyMuPDFLoader(self.filepath)
        documents = loader.load()
                
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # docs = text_splitter.split_documents(documents)

        pattern = r"/([^/]+)$"
        match = re.search(pattern, self.filepath)
        try:
            file_name = match.group(1)
        except:
            file_name = os.path.basename(self.filepath)

        return documents, file_name
        
    

