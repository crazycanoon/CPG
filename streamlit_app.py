import streamlit as st
from newspaper import Article
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# OpenAI API Key (replace with your own key or use Streamlit secrets)
openai.api_key = "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"

st.title("🚀 Web URL Q&A Tool")

# Input section
urls = st.text_area("Enter one or more URLs (comma-separated)")
question = st.text_input("Ask a question based on the content")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

# Function to scrape a single URL
def scrape_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return ""

# Function to split large text into chunks
def chunk_text(text, max_length=500):
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

# Function to create FAISS index
def create_index(chunks):
    vectors = model.encode(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

# Get top matching chunks
def get_top_chunks(question, chunks, index, k=3):
    q_vec = model.encode([question])
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0]]

# Function to generate answer using GPT
def ask_gpt(context, question):
    prompt = f"""Answer the following question using only the information provided below.
    
Context:
{context}

Question: {question}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].strip()

# On button click
if st.button("Get Answer"):
    if not urls or not question:
        st.warning("Please provide both URLs and a question.")
    else:
        with st.spinner("Processing..."):
            url_list = [url.strip() for url in urls.split(",") if url.strip()]
            all_texts = [scrape_url(url) for url in url_list]
            combined_text = "\n".join(all_texts)

            if not combined_text.strip():
                st.error("Failed to extract any content. Check the URLs.")
            else:
                chunks = chunk_text(combined_text)
                index, _ = create_index(chunks)
                top_chunks = get_top_chunks(question, chunks, index)
                context = "\n".join(top_chunks)
                answer = ask_gpt(context, question)
                st.success("Answer:")
                st.write(answer)
