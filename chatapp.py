import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS  # Use langchain_community if you have LangChain >=0.1.0
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ✅ Load environment variables
load_dotenv()

# ✅ Get API Key from .env file
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Google API key not found. Please check your .env file.")
else:
    genai.configure(api_key=api_key)

# ✅ Set FAISS Index Path
FAISS_INDEX_PATH = os.path.join(os.getcwd(), "faiss_index")

# ✅ Set Streamlit Page Config at the top
st.set_page_config(page_title="Multi-PDF Chatbot", page_icon="📚")

# ✅ Function to Extract Text from PDFs
def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ✅ Function to Split Text into Chunks
def get_text_chunks(text):
    """Splits text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# ✅ Function to Create and Store FAISS Vector Database
def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

# ✅ Function to Load Conversational AI Chain
def get_conversational_chain():
    """Loads a question-answering chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available, say "Answer is not available in the context" and do not guess.

    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.6)
  # High accuracy

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ✅ Function to Handle User Input
def user_input(user_question):
    """Handles user queries by searching the FAISS index and generating a response."""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error("FAISS index not found. Please upload and process PDFs first.")
        return
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    # ✅ Fixed Chain Invocation
    response = chain.invoke({"input_documents": docs, "question": user_question})

    st.write("Reply: ", response.get("output_text", "No response generated."))

# ✅ Streamlit UI Setup
def main():
    """Main function to run the Streamlit app."""
    st.header("Multi-PDF Chatbot 📚🤖")

    # ✅ User Question Input
    user_question = st.text_input("Ask a question about the uploaded PDFs... ✍️📝")
    if user_question:
        user_input(user_question)

    # ✅ Sidebar for PDF Upload
    with st.sidebar:
        st.image(os.path.join(os.getcwd(), "img", "Robot.jpg"))
        st.write("---")
        
        st.title("📁 Upload PDF Files")
        pdf_docs = st.file_uploader("Upload PDFs & click Submit to process", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("No PDFs uploaded. Please upload files before processing.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")

        st.write("---")
        st.image(os.path.join(os.getcwd(), "img", "gkj.jpg"))
        st.write("AI App created by @ Gurpreet Kaur")

    # ✅ Footer
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/gurpreetkaurjethra" target="_blank">Gurpreet Kaur Jethra</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
