import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import time

# Fetch OpenAI API key
OPENAI_API_KEY = st.secrets["API_KEY"]
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please check your API key configuration.")

# Extract text from PDF file
def pdf_text_extract(pdf_doc):
    text_list = []
    try:
        pdf_reader = PdfReader(pdf_doc)
        for i in pdf_reader.pages:
            text_list.append(i.extract_text() or "")
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text_list

# Create FAISS index from text data
def create_faiss_index(text_list):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = []  # Store vectors
    texts = []  # Store original texts

    # Convert text to embeddings and prepare for FAISS
    for text in text_list:
        vector = embeddings.embed_query(text)
        vector_store.append(vector)
        texts.append(text)
    
    vector_store = np.array(vector_store).astype(np.float32)

    # Initialize FAISS index
    index = faiss.IndexFlatL2(vector_store.shape[1])
    index.add(vector_store)

    return index, vector_store, texts

# Perform search using FAISS index
def search_data_faiss(index, vector_store, texts, user_question):
    # Create embeddings instance
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Convert the user question to an embedding vector
    query_vector = embeddings.embed_query(user_question)  # Get the embedding for the query
    
    # Perform the search using FAISS
    D, I = index.search(np.array([query_vector]).astype(np.float32), k=10)  # Search for top 10 matches
    
    # Retrieve the text corresponding to the indices
    results = [texts[idx] for idx in I[0]]  # Assuming texts is a list containing the original text
    return results

# Create chain for question answering
def creating_chain():
    Prompt_Template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say: "The answer is not available in the context"\n\n
    Context:\n{context}\n
    Question:\n{user_question}\n

    Answer:
    """
    
    # Initialize model directly, not via kwargs
    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY)
    
    prompt = PromptTemplate(template=Prompt_Template, input_variables=["context", "user_question"])
    
    # Return the chain with the correct initialization
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to ask question and get response from the chatbot
def ask_question(index, vector_store, texts, user_question):
    docs = search_data_faiss(index, vector_store, texts, user_question)
    if not docs:
        st.error("No relevant documents found.")
        return None
    else:
        converted_docs = [Document(page_content=i) for i in docs]
        chain = creating_chain()
        response = chain({"input_documents": converted_docs, "user_question": user_question}, return_only_outputs=True)
        return response["output_text"]

def main():
    st.set_page_config("PDF Chatbot", page_icon="📄", layout="wide")
    st.header("PDF Chatbot 📄")
    st.sidebar.header("Upload your PDF: ")
    
    # File uploader for PDF
    pdf_doc = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
    # Initialize ChromaDB collection
    collection = openai_embedding()

    if pdf_doc is not None:
        # Extract text from the uploaded PDF
        text_list = pdf_text_extract(pdf_doc)
        if not text_list:
            st.error("No text extracted from the uploaded PDF.")
        else:
            # Option to upload data to ChromaDB
            if st.sidebar.button("Upload to ChromaDB"):
                with st.sidebar:
                    with st.spinner("Uploading to ChromaDB..."):
                        time.sleep(3)
                        add_data_to_chromadb(collection, text_list)
                        st.sidebar.success("Documents successfully added to ChromaDB.")
    else:
        st.sidebar.info("Please upload a PDF file to begin.")

    # Check if chat history exists, if not, initialize it
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history in an interactive chat window
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"], unsafe_allow_html=True)
        else:
            st.chat_message("assistant").markdown(message["content"], unsafe_allow_html=True)

    # Get user input and process it
    user_input = st.chat_input("Ask your question:")
    if user_input:
        # Append user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input, unsafe_allow_html=True)

        # Process the user question with a spinner for feedback
        with st.spinner("Processing..."):
            result = ask_question(collection, user_input)  # Get the response
            if result is not None:
                response = result
            else:
                response = "Sorry, I couldn't find any relevant information."

        # Append the assistant's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response, unsafe_allow_html=True)

# Run the main function to start the app
main()

