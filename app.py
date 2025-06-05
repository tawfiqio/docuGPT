import streamlit as st
from dotenv import load_dotenv
from utils import extract_text_from_pdf, chunk_text
from qa import load_vectorstore, get_qa_chain, query_doc

load_dotenv()

st.set_page_config(page_title="DocuGPT", layout="wide")
st.title("📄 DocuGPT – Ask Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask a question about your document:")

if uploaded_file and query:
    with st.spinner("Processing document..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        vectorstore = load_vectorstore(chunks)
        chain = get_qa_chain(vectorstore)
        answer = query_doc(chain, query)
        st.success("Answer:")
        st.write(answer)
