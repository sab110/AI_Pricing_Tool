import os
import faiss
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Set up the path for the data directory and output vector store
data_dir = 'data'  # Directory containing the CSV files
vector_store_path = 'faiss_vector_store_main'  # Path to save the FAISS vector store

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize FAISS index
index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))

# Create an in-memory document store
docstore = InMemoryDocstore()

# Initialize the FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id={}
)

# Function to process all CSV files in the data directory and add to vector store
def process_csv_files_and_store():
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            print(f"Processing file: {file_path}")
            
            # Load and split the CSV file using CSVLoader
            loader = CSVLoader(file_path=file_path,encoding='utf-8-sig')
            docs = loader.load_and_split()
            
            # Add the documents to the vector store
            vector_store.add_documents(documents=docs)
    
    # Save the vector store to disk
    vector_store.save_local(vector_store_path)
    print(f"Vector store saved to {vector_store_path}")

# Call the function to process files and store embeddings
process_csv_files_and_store()
