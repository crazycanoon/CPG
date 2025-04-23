import streamlit as st
from newspaper import Article
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Load API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Page config
st.set_page_config(page_title="Web Q&A App", layout="centered")
st.title("🧠 Ask Questions from Web Pages")

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

try:
    embed_model = SentenceTransformer('./cached_model')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")


# Scrape article content
def scrape_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"[Error scraping {url}]: {str(e)}"

# Break text into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Build FAISS index
def create_index(chunks):
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve relevant chunks
def get_top_chunks(question, chunks, index, top_k=3):
    q_vec = embed_model.encode([question])
    _, indices = index.search(np.array(q_vec), top_k)
    return [chunks[i] for i in indices[0]]

# Ask GPT using limited context
def ask_gpt(context, question):
    prompt = f"""You are a helpful assistant. Use only the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

# Section: Source input
st.markdown("### 🔗 Provide Source URLs")
with st.expander("Click to enter webpage URLs (comma-separated)"):
    urls = st.text_area("Example: https://en.wikipedia.org/wiki/DevOps, https://realpython.com/python-web-scraping/")

# Section: Prompt input
st.markdown("### 💬 Ask Your Question")
question = st.text_input("Type your question based on the content from those pages:")

# Button
if st.button("Get Answer"):
    if not urls.strip() or not question.strip():
        st.warning("Please provide both source URLs and a question.")
    else:
        with st.spinner("Scraping and thinking..."):
            url_list = [u.strip() for u in urls.split(",") if u.strip()]
            scraped_texts = [scrape_url(url) for url in url_list]
            full_text = "\n".join(scraped_texts)
            chunks = chunk_text(full_text)
            index, _ = create_index(chunks)
            top_chunks = get_top_chunks(question, chunks, index)
            context = "\n".join(top_chunks)
            answer = ask_gpt(context, question)

        st.success("✅ Answer")
        st.write(answer)
