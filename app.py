# import streamlit as st
# from retriever import rag_chain  # Assuming the retriever.py is in the same directory
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import os
# # from faiss_retrieverer import rag_chain as faiss_rag_chain

# # Load environment variables
# load_dotenv()

# # Set up OpenAI Embeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))

# def main():
#     # Set up the Streamlit page configuration
#     st.set_page_config(page_title="AI Product Information Retriever", layout="wide")

#     # Adding custom CSS for a sleek and sexy design
#     # Adding custom CSS for a sleek and sexy design
#     st.markdown("""
#         <style>
#         /* Overall background and layout */
#         .stApp {
#             background: linear-gradient(135deg, #ffffff, #d6e5f0);
#             color: #333;
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             margin: 0;
#         }

#         /* Title styling */
#         .stTitle {
#             font-size: 50px;
#             font-weight: bold;
#             color: #0c4b6e;
#             text-align: center;
#             margin-top: 30px;
#             letter-spacing: 1.5px;
#         }

#         /* Input box styling */
#         .stTextInput input {
#             background-color: #f1f5f9;
#             color: #333;
#             font-size: 18px;
#             padding: 12px 20px;
#             border-radius: 50px;
#             border: 1px solid #ccc;
#             box-shadow: 0 4px 10px rgba(0,0,0,0.05);
#             width: 100%;
#             transition: all 0.3s ease;
#         }
#         .stTextInput input:focus {
#             outline: none;
#             border-color: #a0b5d1;
#             box-shadow: 0 0 10px rgba(160, 181, 209, 0.4);
#         }

#         /* Response Box Styling */
#         .response-box {
#             background-color: #ffffff;
#             color: #333;
#             border-radius: 12px;
#             box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
#             padding: 25px;
#             margin-top: 20px;
#             font-size: 18px;
#             max-width: 800px;
#             margin-left: auto;
#             margin-right: auto;
#             line-height: 1.6;
#         }

#         /* Loading Spinner Styling */
#         .stSpinner {
#             color: #4f8dff;
#         }

#         /* Warning Message Styling */
#         .stWarning {
#             font-size: 18px;
#             color: #f36f6f;
#             font-weight: bold;
#             text-align: center;
#             margin-top: 20px;
#         }

#         </style>
#     """, unsafe_allow_html=True)


#     # Title of the app
#     st.title("AI Product Information Retriever")

#     # Add a cool intro section with description
#     st.markdown("""
#         <h3 style="text-align:center; color: #555;">Your AI Assistant for Product Information</h3>
#         <p style="text-align:center; color: #888;">Ask about product details, specifications, pricing, and more from our catalog!</p>
#     """, unsafe_allow_html=True)

#     # Input box to take user query
#     query = st.text_input("Enter your query:", placeholder="e.g., What is the price of Product X?")

#     # Button to trigger retrieval
#     if st.button("Get Answer"):
#         if query:
#             with st.spinner('Retrieving your answer...'):
#                 # Process the query using rag_chain function
#                 persist_directory = "langchain_qdrant_1"  # Path where the Qdrant vector store is located
#                 response = rag_chain(query, persist_directory, embeddings)
#                 # response = faiss_rag_chain(query)
                
                

#                 # Display the response in a cool, clean box
#                 st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
#         else:
#             st.warning("Please enter a query to proceed.", icon="⚠️")


import streamlit as st
from retriever import rag_chain
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import tempfile
from faiss_embeddings import overwrite_vector_store_from_files, append_files_to_existing_vector_store

# Load environment variables
load_dotenv()

# Setup embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store_path = "faiss_vector_store_main"

def main():
    st.set_page_config(page_title="AI Product Information Retriever", layout="wide")

    # Sidebar: Upload & update knowledge base
    st.sidebar.title("📂 Knowledge Base Manager")
    uploaded_files = st.sidebar.file_uploader("Upload CSV or Excel Files", type=["csv"], accept_multiple_files=True)
    update_mode = st.sidebar.radio("Update Mode", ["Append to Vector Store", "Overwrite Vector Store"])
    process_button = st.sidebar.button("📤 Process Files")

    if process_button and uploaded_files:
        with st.spinner("Processing uploaded files..."):
            tmp_paths = []
            for uploaded_file in uploaded_files:
                ext = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_paths.append(tmp.name)

            if update_mode == "Overwrite Vector Store":
                # Save all files in one temp dir and pass the dir
                temp_dir = os.path.dirname(tmp_paths[0])
                overwrite_vector_store_from_files(temp_dir, vector_store_path, embeddings)
            else:
                for path in tmp_paths:
                    append_files_to_existing_vector_store(path, vector_store_path, embeddings)

        st.sidebar.success("Knowledge base updated successfully ✅")

    # Styling
    st.markdown("""<style>/* [your custom CSS here, unchanged] */</style>""", unsafe_allow_html=True)

    # App Title and description
    st.title("AI Product Information Retriever")
    st.markdown("""
        <h3 style="text-align:center; color: #555;">Your AI Assistant for Product Information</h3>
        <p style="text-align:center; color: #888;">Ask about product details, specifications, pricing, and more from our catalog!</p>
    """, unsafe_allow_html=True)

    # User Query
    query = st.text_input("Enter your query:", placeholder="e.g., What is the price of Product X?")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Retrieving your answer..."):
                response = rag_chain(query, vector_store_path, embeddings)
                st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a query to proceed.", icon="⚠️")

if __name__ == "__main__":
    main()
