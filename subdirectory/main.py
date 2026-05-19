def load_urls(urls):
    all_docs=[]
    failed_docs=[]
    for url in urls:
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
            if text:
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": url}
                ))
            else:
                failed_docs.append(url)
        except Exception as e:
            failed_docs.append(url)
            st.warning(f"⚠️ Failed to load: {url} — {e}")
    return all_docs, failed_docs
import os
import trafilatura
import streamlit as st
# import pickle
import time
from langchain_groq import ChatGroq
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever  # FIX 1: multi-query retriever
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url.strip())
urls = [u.strip() for u in urls if u.strip()]
# FIX 2: deduplicate URLs so we don't triple-embed the same article
urls = list(dict.fromkeys(urls))

process_url_clicked = st.sidebar.button("Process URLs")

# FIX 3: temperature=0 for factual retrieval (was 0.6 → hallucination risk)
llm = ChatGroq(
    api_key=key,
    model="llama-3.3-70b-versatile",
    temperature=0
)

@st.cache_resource # 3
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
embedding_model = load_embedding_model()
status_placeholder = st.empty() # 5

if process_url_clicked:
    if not urls:
        st.error("Please enter at least one URL before processing.")
        st.stop()
 
    # FIX 4: load URLs individually so we can report per-URL failures
    all_docs, failed_urls = load_urls(urls)
 
    if not all_docs:
        st.error("No content could be loaded from any of the provided URLs.")
        st.stop()

    status_placeholder.text("Data Loading...Done ✅✅✅")
    
    # split data
    # FIX 6: chunk_overlap=200 so context isn't lost at chunk boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200
    )
    status_placeholder.text("Text Splitting...Done ✅✅✅")
    docs = text_splitter.split_documents(all_docs)
    
    if not docs:
        st.error("No documents created after splitting")
        st.stop()

    # FIX 7: store vector store in session_state, not on disk
    # This prevents users on a shared deployment from loading each other's indexes
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
    )
    st.session_state["vector_store"] = vector_store
    status_placeholder.text(f"Embeddings built from {len(docs)} chunks across {len(all_docs)} article(s) ✅✅️✅️")
    time.sleep(1)
    status_placeholder.empty()
    st.success(f"Ready! Loaded {len(all_docs)} articles.")

# FIX 8: read vector store from session_state, not from disk
if "vector_store" in st.session_state:
    vectorStore = st.session_state["vector_store"]

    # FIX 9: MultiQueryRetriever — decomposes compound questions into sub-queries
    # This is the core fix for question containing two contexts
    # It generates multiple search queries from your question, runs each separately,
    # then merges all retrieved chunks before passing to the LLM.
    base_retriever = vectorStore.as_retriever(
        search_kwargs={"k": 8}  # FIX 10: increased k from default 4 → 8
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=multi_query_retriever
    )
    
    query = st.text_input("Question: ")
    if query:
        with st.spinner("Searching articles..."):
            result = chain({"question": query}, return_only_outputs=True)
        
        # {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result.get("answer", "No answer found."))

        # FIX 11: sources is comma-separated, not newline-separated
        # Display sources if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = [source.strip() for source in sources.split(",") if source.strip()] # split the sources by newline
            for source in sources_list:
                st.write(source)
else:
    st.info("👈 Paste article URLs in the sidebar and click **Process URLs** to get started.")
    
