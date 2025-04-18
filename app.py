import streamlit as st
from retriever import rag_chain  # Assuming the retriever.py is in the same directory
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
# from faiss_retrieverer import rag_chain as faiss_rag_chain

# Load environment variables
load_dotenv()

# Set up OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))

def main():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="AI Product Information Retriever", layout="wide")

    # Adding custom CSS for a sleek and sexy design
    # Adding custom CSS for a sleek and sexy design
    st.markdown("""
        <style>
        /* Overall background and layout */
        .stApp {
            background: linear-gradient(135deg, #ffffff, #d6e5f0);
            color: #333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
        }

        /* Title styling */
        .stTitle {
            font-size: 50px;
            font-weight: bold;
            color: #0c4b6e;
            text-align: center;
            margin-top: 30px;
            letter-spacing: 1.5px;
        }

        /* Input box styling */
        .stTextInput input {
            background-color: #f1f5f9;
            color: #333;
            font-size: 18px;
            padding: 12px 20px;
            border-radius: 50px;
            border: 1px solid #ccc;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            width: 100%;
            transition: all 0.3s ease;
        }
        .stTextInput input:focus {
            outline: none;
            border-color: #a0b5d1;
            box-shadow: 0 0 10px rgba(160, 181, 209, 0.4);
        }

        /* Response Box Styling */
        .response-box {
            background-color: #ffffff;
            color: #333;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            padding: 25px;
            margin-top: 20px;
            font-size: 18px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        /* Loading Spinner Styling */
        .stSpinner {
            color: #4f8dff;
        }

        /* Warning Message Styling */
        .stWarning {
            font-size: 18px;
            color: #f36f6f;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }

        </style>
    """, unsafe_allow_html=True)


    # Title of the app
    st.title("AI Product Information Retriever")

    # Add a cool intro section with description
    st.markdown("""
        <h3 style="text-align:center; color: #555;">Your AI Assistant for Product Information</h3>
        <p style="text-align:center; color: #888;">Ask about product details, specifications, pricing, and more from our catalog!</p>
    """, unsafe_allow_html=True)

    # Input box to take user query
    query = st.text_input("Enter your query:", placeholder="e.g., What is the price of Product X?")

    # Button to trigger retrieval
    if st.button("Get Answer"):
        if query:
            with st.spinner('Retrieving your answer...'):
                # Process the query using rag_chain function
                persist_directory = "langchain_qdrant_1"  # Path where the Qdrant vector store is located
                response = rag_chain(query, persist_directory, embeddings)
                # response = faiss_rag_chain(query)
                
                

                # Display the response in a cool, clean box
                st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a query to proceed.", icon="⚠️")

if __name__ == "__main__":
    main()
