import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Web Q&A (Cloud-Friendly)", layout="wide")
st.title("🔎 Ask Questions from Webpages")

HF_API_TOKEN = st.secrets["api"]["hf_token"]
HF_EMBEDDING_ENDPOINT = "https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2"

def embed_sentences_hf(sentences):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {"inputs": sentences}
    response = requests.post(HF_EMBEDDING_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    return np.array(response.json())

@st.cache_resource
def load_qa_model():
    try:
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    except Exception as e:
        st.error(f"❌ QA model load error: {e}")
        st.stop()

qa_model = load_qa_model()

with st.expander("🔗 Source URLs"):
    urls_input = st.text_area("Enter URLs (one per line)", height=150)

with st.expander("❓ Your Question"):
    user_question = st.text_input("Enter your question")

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

if st.button("💡 Get Answer"):
    if not urls_input.strip() or not user_question.strip():
        st.warning("Enter URLs and a question.")
    else:
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        full_text = "\n".join([scrape_text(u) for u in urls])
        chunks = chunk_text(full_text)

        if not chunks:
            st.error("No readable text found.")
        else:
            try:
                st.info("Embedding text via Hugging Face API...")
                embeddings = embed_sentences_hf(chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)

                question_vec = embed_sentences_hf([user_question])
                D, I = index.search(question_vec, k=3)

                context = "\n".join([chunks[i] for i in I[0]])
                result = qa_model(question=user_question, context=context)

                st.success("Answer:")
                st.write(result["answer"])
            except Exception as e:
                st.error(f"Processing error: {e}")
