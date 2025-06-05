import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def load_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_texts(chunks, embeddings)

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def query_doc(chain, query):
    return chain.run(query)
