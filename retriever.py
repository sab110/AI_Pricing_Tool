# from langchain_qdrant import QdrantVectorStore
# from langchain.chat_models import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# from qdrant_client import QdrantClient
# from dotenv import load_dotenv
# import os
# import re
# import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)
# def clean_text(text):
#     """Remove special characters like ** and # from the text."""
#     text = re.sub(r'[\*\#]', '', text)  # Removes asterisks (*) and hashtags (#)
#     return text.strip()  # Ensures there are no leading or trailing spaces

# # # Global dictionary to store chat history
# # chat_history = {}


# # Load environment variables
# load_dotenv()

# def load_vector_db(persist_directory, collection_name, embeddings):
#     client = QdrantClient(path=persist_directory)
#     vectorstore = QdrantVectorStore(
#         client=client,
#         collection_name=collection_name,
#         embedding=embeddings,
#     )
#     retriever = vectorstore.as_retriever()
#     return retriever

# def combine_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# def rag_chain(question, persist_directory, embeddings):
#     retriever = load_vector_db(persist_directory, collection_name="demo_collection", embeddings=embeddings)
#     retrieved_docs = retriever.get_relevant_documents(question,k=1)

#     if not retrieved_docs:
#         print("No documents retrieved.")
#         return ""
#     else:
#         print(f"Retrieved {len(retrieved_docs)} documents.")
#         # print("Documents retrieved:", retrieved_docs)

#     # formatted_context = combine_docs(retrieved_docs)
#     response = generate_response(retrieved_docs, question)
    
#     return response

# def generate_response(context, question, temperature=0.7):
#     llm = ChatOpenAI(
#         temperature=temperature,
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         model="gpt-4o-mini",
#         max_tokens=200
#     )

#     formatted_prompt = f"""
#     Context:
#     {context}

#     Question:
#     {question}

#     Provide a concise, accurate, and well-structured response based on the context above, ensuring proper grammar, clarity, and coherence:    """
#     return llm.predict(formatted_prompt)






# def main(query):
#     persist_directory = "langchain_qdrant"
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
#     # question = input("Enter your query: ")
#     result = rag_chain(query, persist_directory, embeddings)
    
#     clean_result = clean_text(result.strip())  
#     print("\nResponse:\n" + "-" * 50)  
#     print(clean_result)  
#     print("-" * 50)  




# # question = input("Enter your query: ")
# # main(question)




from langchain_qdrant import QdrantVectorStore
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import re
import warnings
from langchain_community.vectorstores import FAISS
warnings.filterwarnings("ignore", category=DeprecationWarning)

def clean_text(text):
    """Remove special characters like ** and # from the text."""
    text = re.sub(r'[\*\#]', '', text)  # Removes asterisks (*) and hashtags (#)
    return text.strip()  # Ensures there are no leading or trailing spaces

# Load environment variables
load_dotenv()

def load_vector_db(embeddings):
    # client = QdrantClient(path=persist_directory)
    # vectorstore = QdrantVectorStore(
    #     client=client,
    #     collection_name=collection_name,
    #     embedding=embeddings,
    # )
    # Load the saved FAISS vector store
    vector_store_path = 'faiss_vector_store_main'  # Path to the saved FAISS vector store
    vector_store = FAISS.load_local(
        vector_store_path, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever()
    return retriever

# def combine_docs(docs, max_words=400):
#     """Combine documents but limit the total word count to max_words."""
#     combined = ""
#     word_count = 0
    
#     for doc in docs:
#         doc_words = doc.page_content.split()
#         if word_count + len(doc_words) > max_words:
#             # Only add the remaining words to meet the max word limit
#             remaining_words = max_words - word_count
#             combined += " ".join(doc_words[:remaining_words])
#             break
#         else:
#             combined += doc.page_content + "\n\n"
#             word_count += len(doc_words)
    
#     return combined

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question, persist_directory, embeddings):
    # retriever = load_vector_db(persist_directory, collection_name="demo_collection", embeddings=embeddings)
    retriever = load_vector_db(embeddings=embeddings)
    
    retrieved_docs = retriever.get_relevant_documents(question, k=5)

    if not retrieved_docs:
        print("No documents retrieved.")
        return ""
    else:
        print(f"Retrieved {len(retrieved_docs)} documents.")
        # print(retrieved_docs)

    # Limit the context to 400 words
    # context = combine_docs(retrieved_docs, max_words=100000)
    context = combine_docs(retrieved_docs)
    response = generate_response(context, question)
    
    return response

def generate_response(context, question, temperature=0.9):
    llm = ChatOpenAI(
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        
    )

    formatted_prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Please compare the question with the context provided and provide a concise, accurate, and well-structured response.
    """

    print(formatted_prompt)  # Debugging line to check the formatted prompt
    return llm.predict(formatted_prompt)

def main(query):
    persist_directory = "langchain_qdrant_1"
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    result = rag_chain(query, persist_directory, embeddings)
    
    clean_result = clean_text(result.strip())  
    print("\nResponse:\n" + "-" * 50)  
    print(clean_result)  
    print("-" * 50)

# if __name__ == "__main__":
#     query = input("Enter your query: ")
#     main(query)
