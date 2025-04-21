import os
import faiss
import pandas as pd
from langchain_community.document_loaders import UnstructuredExcelLoader, CSVLoader
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

# # Function to process all CSV files in the data directory and add to vector store
# def process_csv_files_and_store():
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(data_dir, filename)
#             print(f"Processing file: {file_path}")
            
#             # Load and split the CSV file using CSVLoader
#             loader = CSVLoader(file_path=file_path,encoding='utf-8-sig')
#             docs = loader.load_and_split()
            
#             # Add the documents to the vector store
#             vector_store.add_documents(documents=docs)
    
#     # Save the vector store to disk
#     vector_store.save_local(vector_store_path)
#     print(f"Vector store saved to {vector_store_path}")

# # Call the function to process files and store embeddings
# process_csv_files_and_store()



def load_documents_from_file(file_path: str):
    """
    Dynamically loads documents from CSV or Excel files.

    Args:
        file_path (str): The full path to the file.
    
    Returns:
        list: List of loaded and split documents.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        loader = CSVLoader(file_path=file_path, encoding='utf-8-sig')
        return loader.load_and_split()
    elif ext in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(file_path=file_path, mode="elements")
        return loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")



def overwrite_vector_store_from_files(data_dir: str, vector_store_path: str, embeddings: OpenAIEmbeddings):
    """
    Overwrites the FAISS vector store using all supported files in the directory.

    Args:
        data_dir (str): Path to the directory with files.
        vector_store_path (str): Path to save the FAISS store.
        embeddings (OpenAIEmbeddings): Embedding model.
    """
    index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))
    docstore = InMemoryDocstore()
    vector_store = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id={})

    for filename in os.listdir(data_dir):
        if filename.endswith(('.csv', '.xlsx', '.xls')):
            file_path = os.path.join(data_dir, filename)
            print(f"Processing: {file_path}")
            docs = load_documents_from_file(file_path)
            vector_store.add_documents(docs)

    vector_store.save_local(vector_store_path)
    print(f"Vector store overwritten and saved to: {vector_store_path}")
    
def append_files_to_existing_vector_store(file_path_or_dir: str, vector_store_path: str, embeddings: OpenAIEmbeddings):
    """
    Appends supported file(s) from a file or directory to an existing FAISS vector store.

    Args:
        file_path_or_dir (str): Path to a CSV/Excel file or a directory containing such files.
        vector_store_path (str): Path where the FAISS vector store is saved.
        embeddings (OpenAIEmbeddings): Embedding model instance.
    """
    print(f"Loading vector store from: {vector_store_path}")
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

    if os.path.isfile(file_path_or_dir):
        # Handle single file
        ext = os.path.splitext(file_path_or_dir)[1].lower()
        if ext in [".csv", ".xlsx", ".xls"]:
            print(f"Appending file: {file_path_or_dir}")
            docs = load_documents_from_file(file_path_or_dir)
            vector_store.add_documents(docs)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    elif os.path.isdir(file_path_or_dir):
        # Handle directory
        for filename in os.listdir(file_path_or_dir):
            if filename.endswith((".csv", ".xlsx", ".xls")):
                full_path = os.path.join(file_path_or_dir, filename)
                print(f"Appending file: {full_path}")
                docs = load_documents_from_file(full_path)
                vector_store.add_documents(docs)
    else:
        raise FileNotFoundError(f"Path does not exist: {file_path_or_dir}")

    vector_store.save_local(vector_store_path)
    print(f"Updated vector store saved to: {vector_store_path}")


# if __name__ == "__main__":
#     # Overwrite entire store
#     overwrite_vector_store_from_files(data_dir, vector_store_path, embeddings)
#     # append_files_to_existing_vector_store('data', vector_store_path, embeddings)