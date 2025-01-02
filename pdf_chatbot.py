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

OPENAI_API_KEY = st.secrets["API_KEY"]
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please check your API key configuration.")


def pdf_text_extract(pdf_doc):
    text_list = []
    try:
        pdf_reader = PdfReader(pdf_doc)
        for i in pdf_reader.pages:
            text_list.append(i.extract_text() or "")
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text_list


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

def creating_chain():
    Prompt_Template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say: "The answer is not available in the context"\n\n
    Context:\n{context}\n
    Question:\n{user_question}\n

    Answer:
    """
    
    # Pass the model directly as an argument, not through kwargs
    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY)
    
    prompt = PromptTemplate(template=Prompt_Template, input_variables=["context", "user_question"])
    
    # Return the chain with the correct initialization
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)





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
    st.set_page_config("pdf chatbot")
    st.header("PDF Chatbot 📄")
    st.sidebar.header("Upload your Pdf: ")
    pdf_doc = st.sidebar.file_uploader("", type="pdf")
    
    if pdf_doc is not None:
        text_list = pdf_text_extract(pdf_doc)
        if not text_list:
            st.error("No text extracted from the uploaded PDF.")
        else:
            # Create FAISS index and store vectors and texts
            index, vector_store, texts = create_faiss_index(text_list)

            if st.sidebar.button("Upload to FAISS"):
                with st.sidebar:
                    with st.spinner("Uploading"):
                        time.sleep(3)
                        st.sidebar.success("Documents successfully added to FAISS.")

    else:
        st.sidebar.info("Please upload a PDF file to begin.")

    if "chat_history" not in st.session_state:       # creating chat history in session state
        st.session_state.chat_history = []

    for i in st.session_state.chat_history:          # iterating to chat_histroy to display all the chat
        st.chat_message(i["role"]).write(i["content"])

    user_input = st.chat_input("Ask your question:")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})  # appending user message to the chat_history
        st.chat_message("user").write(user_input)

        with st.spinner("Processing..."):
            result = ask_question(index, vector_store, texts, user_input)
            if result is not None:
                response = result
            else:
                st.error("No relevant answer found in the uploaded documents.")

        st.session_state.chat_history.append({"role": "assistant", "content": response})  # Appending bot response to the chat history 
        st.chat_message("assistant").write(response)

main()
