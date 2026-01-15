from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_dir: str = "chroma_db"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Explicit for reproducibility
        )
    
    def build_and_save_vectorstore(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        loader = CSVLoader(
            file_path=self.csv_path,
            encoding='utf-8',
            metadata_columns=['Name', 'Genres'],  # Preserve key fields
            csv_args={'quoting': 1}  # Handle quoted fields better
        )
        data = loader.load()
        
        if not data:
            raise ValueError("No data loaded from CSV")
        
        # Anime-optimized chunking: smaller chunks with overlap for better retrieval
        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\nTitle: ",
            keep_separator=True
        )
        texts = splitter.split_documents(data)
        
        db = Chroma.from_documents(
            texts, 
            self.embedding, 
            persist_directory=self.persist_dir
        )
        print(f"Vector store created with {db._collection.count()} documents")
        return db
    
    def load_vector_store(self):
        return Chroma(
            persist_directory=self.persist_dir, 
            embedding_function=self.embedding
        )