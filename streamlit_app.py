import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Web Q&A", layout="wide")
st.title("🔎 Ask Questions from Webpages (For 100th time)")

# URL input section
with st.expander("🔗 Source URLs"):
    urls_input = st.text_area("Enter one or more URLs (one per line)", height=150)

# Prompt input section
with st.expander("❓ Ask a Question"):
    user_question = st.text_input("Enter your question here:")

@st.cache_resource
def load_embedder():
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", trust_remote_code=False)
        model.encode(["test"], show_progress_bar=False)  # Force model initialization
        return model
    except Exception as e:
        st.error(f"❌ Failed to load embedding model.\n\nError: {e}")
        st.stop()

@st.cache_resource
def load_qa_model():
    try:
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
    except Exception as e:
        st.error(f"❌ Failed to load QA model.\n\nError: {e}")
        st.stop()

embed_model = load_embedder()
qa_model = load_qa_model()

def scrape_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        st.warning(f"Could not fetch {url}: {e}")
        return ""

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

if st.button("💡 Get Answer"):
    if not urls_input.strip() or not user_question.strip():
        st.warning("Please enter both URLs and a question.")
    else:
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        full_text = "\n".join([scrape_text(u) for u in urls])
        chunks = chunk_text(full_text)

        if not chunks:
            st.error("Couldn't extract readable text from the URLs.")
        else:
            faiss_index, all_embeddings = build_faiss_index(chunks)
            question_embedding = embed_model.encode([user_question], show_progress_bar=False)
            D, I = faiss_index.search(np.array(question_embedding), k=3)

            context = "\n".join([chunks[i] for i in I[0]])
            with st.spinner("Searching for answer..."):
                try:
                    result = qa_model(question=user_question, context=context)
                    st.success("Answer:")
                    st.write(result["answer"])
                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")
