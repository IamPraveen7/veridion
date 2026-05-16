import os
import streamlit as st
# import pickle
import time
from langchain_groq import ChatGroq
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url.strip())
urls = [u.strip() for u in urls if u.strip()] # 4


process_url_clicked = st.sidebar.button("Process URLs")
llm = ChatGroq(
    api_key=key,
    model="llama-3.3-70b-versatile",
    temperature=0.6
)

@st.cache_resource # 3
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
embedding_model = load_embedding_model()

status_placeholder = st.empty() # 5
if process_url_clicked:
    # load data
    loader = WebBaseLoader(urls)
    status_placeholder.text("Data Loading...Started...✅️✅️✅️")
    data = loader.load()
    if not data:
        st.error("No data loaded from URLs")
        st.stop()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000
    )
    status_placeholder.text("Text Splitting...Started...✅️✅️✅️")
    docs = text_splitter.split_documents(data)
    if not docs:
        st.error("No documents created after splitting")
        st.stop()
    # create embeddings and save to FAISS index
    if embedding_model is not None:
        print("Embeddings Loaded...")
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
    )
    status_placeholder.text("Embedding Vectors Started Building...Started...✅️✅️✅️")
    time.sleep(2)

    # Save th FAISS index to a pickle file
    flag = vector_store.save_local("faiss_index")

if os.path.exists('faiss_index'):
    vectorStore = FAISS.load_local(
        "faiss_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever()
    )
    query = st.text_input("Question: ")
    if query:
        result = chain({"question": query}, return_only_outputs=True)
        # {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result.get("answer"))

        # Display sources if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # split the sources by newline
            for source in sources_list:
                st.write(source)