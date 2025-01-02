import streamlit as st
from PyPDF2 import PdfReader

import faiss
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import time

OPENAI_API_KEY = st.secrets("API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please check your API key configuration.")

# Initialize FAISS index globally
faiss_index = None
embedding_dim = 1536  # Dimensionality of OpenAI embeddings

def pdf_text_extract(pdf_doc):
    text_list = []
    try:
        pdf_reader = PdfReader(pdf_doc)
        for i in pdf_reader.pages:
            text_list.append(i.extract_text() or "")
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text_list

def openai_embedding():
    # Initialize OpenAI Embedding model
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # Using OpenAI embeddings
    openai_emb = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002")
    return openai_emb

def create_faiss_index():
    global faiss_index
    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(embedding_dim)
    return faiss_index

def add_data_to_faiss(embedding_function, text_list):
    global faiss_index
    embeddings = []
    for text in text_list:
        embedding = embedding_function.embed_query(text)
        embeddings.append(embedding)
    
    # Convert to numpy array for FAISS
    embeddings_np = np.array(embeddings).astype(np.float32)
    
    faiss_index.add(embeddings_np)  # Add embeddings to FAISS index

def search_data_faiss(user_question, embedding_function):
    # Get embedding for the user question
    question_embedding = embedding_function.embed_query(user_question)
    question_embedding_np = np.array([question_embedding]).astype(np.float32)

    # Perform a similarity search
    D, I = faiss_index.search(question_embedding_np, 10)  # search for top 10
    return I

def creating_chain():
    Prompt_Template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say: "The answer is not available in the context"\n\n
    Context:\n{context}\n
    Question:\n{user_question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(template=Prompt_Template, input_variables=["context", "user_question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def ask_question(collection, user_question, embedding_function):
    # Find closest documents in FAISS index
    doc_indices = search_data_faiss(user_question, embedding_function)
    
    # Retrieve the documents based on the indices
    docs = [collection[i] for i in doc_indices[0]]  # Extract the corresponding documents
    
    if not docs:
        st.error("No relevant documents found.")
        return None
    else:
        converted_docs = [Document(page_content=i) for i in docs]
        chain = creating_chain()
        response = chain({"input_documents": converted_docs, "user_question": user_question}, return_only_outputs=True)
        return response["output_text"]

def main():
    st.set_page_config("pdf chatbot")
    st.header("PDF Chatbot 📄")
    st.sidebar.header("Upload your Pdf: ")
    pdf_doc = st.sidebar.file_uploader("", type="pdf")
    
    embedding_function = openai_embedding()
    create_faiss_index()

    if pdf_doc is not None:
        text_list = pdf_text_extract(pdf_doc)
        if not text_list:
            st.error("No text extracted from the uploaded PDF.")
        else:
            if st.sidebar.button("Upload to FAISS"):
                with st.sidebar:
                    with st.spinner("Uploading"):
                        time.sleep(3)
                        add_data_to_faiss(embedding_function, text_list)
                        st.sidebar.success("Documents successfully added to FAISS.")
    else:
        st.sidebar.info("Please upload a PDF file to begin.")

    if "chat_history" not in st.session_state:  # creating chat history in session state
        st.session_state.chat_history = []

    for i in st.session_state.chat_history:  # iterating to chat_history to display all the chat
        st.chat_message(i["role"]).write(i["content"])

    user_input = st.chat_input("Ask your question:")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})  # appending user message to the chat_history
        st.chat_message("user").write(user_input)

        with st.spinner("Processing..."):
            result = ask_question(text_list, user_input, embedding_function)
            if result is not None:
                response = result
            else:
                st.error("No relevant answer found in the uploaded documents.")

        st.session_state.chat_history.append({"role": "assistant", "content": response})  # Appending bot response to the chat history
        st.chat_message("assistant").write(response)

main()
