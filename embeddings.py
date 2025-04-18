from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_csv(file_path):
    """
    Extract all text from the CSV file by concatenating all rows.
    """
    df = pd.read_csv(file_path,encoding='utf-8-sig')
    # Concatenate specific columns into a single string (adjust as necessary)
    # document_text = df.astype(str).agg(' '.join, axis=1)  # Combine all columns for each row into a string.
    
    # return document_text  # This will be a Series, which we can pass to the text splitter
    return "\n".join(df.astype(str).agg(' '.join, axis=1))
    # return df

def main():
    # Step 1: Set up embeddings and Qdrant client
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
    client = QdrantClient(path="langchain_qdrant_1")
    
    # Step 2: Create Qdrant collection if it doesn't exist
    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # Step 3: Loop through the CSV files in the "data" folder
    data_folder = "data"
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            print(f"Processing file: {file_path}")

            # Step 4: Extract text from CSV
            document_text = extract_text_from_csv(file_path)
            print(f"Text extracted from {filename}:")
            print(document_text[:200])  # print the first 200 characters for preview

            # Step 5: Chunk the text
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            chunks = text_splitter.split_text(document_text)
            print(f"Text chunks for {filename}: {chunks[:2]}")  # print first two chunks for preview

            # Step 6: Store embeddings in Qdrant
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="demo_collection",
                embedding=embeddings,
            )
            
            # Add chunks to the vector store
            vector_store.add_texts(texts=chunks)

            print(f"Embeddings for {filename} stored successfully!\n")

if __name__ == "__main__":
    main()


# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def main():
#     # Step 1: Set up embeddings and Qdrant client
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
#     client = QdrantClient(path="langchain_qdrant_1")
    
#     # Step 2: Create Qdrant collection if it doesn't exist
#     client.create_collection(
#         collection_name="demo_collection",
#         vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
#     )

#     # Step 3: Loop through the CSV files in the "data" folder
#     data_folder = "data"
#     for filename in os.listdir(data_folder):
#         if filename.endswith(".csv"):
#             file_path = os.path.join(data_folder, filename)
#             print(f"Processing file: {file_path}")
            
#             # Step 4: Use CSVLoader to load the CSV file
#             loader = CSVLoader(file_path=file_path,encoding='utf-8-sig')
#             documents = loader.load()

#             # Step 5: Extract text from CSV (join all document text into one string)
#             # Access the text directly from the Document object
#             document_text = "\n".join([doc.page_content for doc in documents])
#             print(f"Text extracted from {filename}:")
#             print(document_text[:200])  # print the first 200 characters for preview

#             # Step 6: Chunk the text
#             text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#             chunks = text_splitter.split_text(document_text)
#             print(f"Text chunks for {filename}: {chunks[:2]}")  # print first two chunks for preview

#             # Step 7: Store embeddings in Qdrant
#             vector_store = QdrantVectorStore(
#                 client=client,
#                 collection_name="demo_collection",
#                 embedding=embeddings,
#             )
            
#             # Add chunks to the vector store
#             vector_store.add_texts(texts=chunks)

#             print(f"Embeddings for {filename} stored successfully!\n")

# if __name__ == "__main__":
#     main()
