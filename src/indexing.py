import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("unstructured").setLevel(logging.CRITICAL)

import os
import glob
import torch
import shutil
import argparse
from pypandoc import download_pandoc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Indexing:
    """
    Load the documents and split them into chunks with overlap.
    Save the chunks in a vector database.

    Args:
        source_dir(str): The file path to the directory containing your documents.
        vector_path(str): The file path to store the embeddings.
        file_type(str): File type of the documents you wish to be loaded.
        c_size(int): Maximum size of each chunk to return. Chooses 1000 by default.
        c_overlap(int): Maximum overlap between chunks. Chooses 250 by default.
        embedding_model(str): Defines the embedding model to be used on the chunks. Chooses "all-MiniLM-L6-v2" by default.
    """

    def __init__(self, source_dir: str, vector_path: str, file_type: str, c_size: int = 1000, c_overlap: int = 250, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.source_dir = source_dir
        self.vector_path = vector_path
        self.file_type = file_type
        self.c_size = c_size
        self.c_overlap = c_overlap
        self.embedding_model = embedding_model

        self.documents = []
        self.chunks = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __clear_embedding(self):
        """
        Clears the existing vector store if it exists already and creates a new one.
        """
        if os.path.exists(self.vector_path):
            print(f"Clearing old vector store at: {self.vector_path}")
            shutil.rmtree(self.vector_path)
        else:
            print(f"No old vector store found at {self.vector_path}. Creating a new one!")

    def __document_loader(self):
        """
        Loads documents into memory to be fed into the splitter/chunkify function. 
        Currently handles only .rst documents.
        """
        
        supported_types = [".rst"]
        rst_types = ['.rst', '.RST', 'RST', 'rst']
        if self.file_type in rst_types:
            loader = DirectoryLoader(
                self.source_dir, 
                glob = "**/*.rst", 
                loader_cls = UnstructuredFileLoader,
                show_progress=True,
                use_multithreading = True,
                silent_errors=True
            )
            self.documents = loader.load()
            print(f"Successfully loaded {len(self.documents)}.")
        else:
            raise ValueError(
                f"Unsupported file type: {self.file_type}."
                f"This loader currently only supports: {supported_types}"
            )

    def __text_splitter(self):
        """
        Splits the documents into chunks based on the chunk and overlap size.
        """

        if not self.documents:
            print("No documents to split.")
            return 
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.c_size,
            chunk_overlap = self.c_overlap,
            add_start_index = True
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Split all the documents into {len(self.chunks)} chunks")

    def __embed_and_store(self):
        """
        Uses an embedding model to convert each text chunk into vectors that represent its semantic meaning. 
        By defualt, "sentence-transformers/all-MiniLM-L6-v2" is used as the embedding model, but you can choose another embedding model of your choice.
        Stores the embeddings using ChromaDB.
        """

        hf_embedding_model = HuggingFaceEmbeddings(
            model_name = self.embedding_model,
            model_kwargs = {'device': self.device}
        )
        print(f"Embedding model successfully loaded! Using device: {self.device}.")

        db = Chroma.from_documents(
            self.chunks,
            hf_embedding_model,
            persist_directory = self.vector_path
        )

    def pipeline_execute(self):
        """
        Invokes the helper functions in the order of execution of the pipeline. Loading -> Chunkify -> Embedding
        """ 
        try:
            print("Checking for pandoc dependency...")
            download_pandoc()   
            print("Starting the indexing pipeline...")
            self.__clear_embedding()
            self.__document_loader()
            self.__text_splitter()
            self.__embed_and_store() 
            print(f"Indexing completed. Embeddings successfully stored at: {self.vector_path}!")

        except Exception as e:
            print(f"An error occurred while indexing: {e}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=str, help='The file path to the directory containing your documents')
    parser.add_argument("vector_path", type=str, help='The file path to store the embeddings')
    parser.add_argument("file_type", type=str, help="File type of the documents you wish to be loaded")
    parser.add_argument("-cs", "--c_size", type=int, help='Maximum size of each chunk to return', default=1000)
    parser.add_argument("-co", "--c_overlap", type=int, help='Maximum overlap between chunks.', default=250)
    parser.add_argument("-em", "--embedding_model", type=str, help='Defines the embedding model to be used on the chunks.', default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    indexer = Indexing(
        source_dir = args.source_dir,
        vector_path = args.vector_path,
        file_type = args.file_type,
        c_size = args.c_size,
        c_overlap = args.c_overlap,
        embedding_model = args.embedding_model
    )

    indexer.pipeline_execute()