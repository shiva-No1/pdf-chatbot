import streamlit as st
from PyPDF2 import PdfReader

from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_function

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


def openai_embedding():
    openai_emb = embedding_function.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002"
    )
    client = PersistentClient(path="chroma_vector_data/")
    collection = client.get_or_create_collection(name="pdf_collection", embedding_function=openai_emb)
    return collection


def add_data_to_chromadb(collection, text_list):
    doc_ids = [str(i) for i in range(1, len(text_list) + 1)]
    collection.upsert(documents=text_list, ids=doc_ids)
    


def search_data_chromadb(collection, user_question):
    results = collection.query(query_texts=[user_question], n_results= 10)
    return results.get("documents", [[]])[0]


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


def ask_question(collection, user_question):
    docs = search_data_chromadb(collection, user_question)
    if not docs:
        st.error("No relevant documents found.")
        return None
    else:
        converted_docs = [Document(page_content=i) for i in docs]
        chain = creating_chain()
        response = chain({"input_documents": converted_docs, "user_question": user_question}, return_only_outputs = True)
        return response["output_text"]

def main():
    st.set_page_config("pdf chatbot")
    st.header("PDF Chatbot 📄")
    st.sidebar.header("Upload your Pdf: ")
    pdf_doc = st.sidebar.file_uploader("",type="pdf")
    collection = openai_embedding()

    if pdf_doc is not None:
        text_list = pdf_text_extract(pdf_doc)
        if not text_list:
            st.error("No text extracted from the uploaded PDF.")
        else:
            if st.sidebar.button("Upload to ChromaDB"):
                with st.sidebar:
                    with st.spinner("Uploading"):
                        time.sleep(3)
                        add_data_to_chromadb(collection, text_list)
                        st.sidebar.success("Documents successfully added to ChromaDB.")
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
                result = ask_question(collection, user_input)
                if result is not None:
                    response = result
                else:
                    st.error("No relevant answer found in the uploaded documents.")
 
        st.session_state.chat_history.append({"role": "assistant", "content": response})  # Appending bot response to the chat history 
        st.chat_message("assistant").write(response)

main()
