import streamlit as st
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os

# Set your OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Load a pre-trained sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Function to scrape text content from a URL
def scrape_text(url):
    try:
        response = requests.get(url, timeout=10)  # Add a timeout
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract main text content (you might need to adjust selectors based on website structure)
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error processing URL: {e}"

# Function to embed text
def embed_text(text):
    return embedding_model.encode(text)

# Function to perform vector search
def search_text(query_embedding, embeddings, documents, top_n=3):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    sorted_indices = np.argsort(similarities, axis=0)[::-1]
    results = [(documents[i], similarities[i]) for i in sorted_indices[:top_n]]
    return results

# Function to answer the question using OpenAI API
def answer_question_openai(question, context):
    combined_context = "\n\n".join([doc for doc, _ in context])
    prompt = f"Based on the following information:\n\n{combined_context}\n\nAnswer the question: {question}"
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # Or another suitable model
            prompt=prompt,
            max_tokens=200,  # Adjust as needed
            n=1,
            stop=None,
            temperature=0.2,  # Adjust for creativity vs. accuracy
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"Error communicating with OpenAI: {e}"

# Streamlit UI
st.title("Web Content Q&A Tool Powered by OpenAI")

urls_input = st.text_area("Enter one or more URLs (one per line):")
question = st.text_input("Ask a question about the content:")

if st.button("Process URLs and Ask"):
    if urls_input and question:
        urls = [url.strip() for url in urls_input.splitlines() if url.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            st.info("Processing URLs and extracting content...")
            documents = []
            embeddings_list = []
            for url in urls:
                with st.spinner(f"Scraping content from {url}"):
                    text_content = scrape_text(url)
                    if "Error" not in text_content:
                        documents.append(text_content)
                        embeddings_list.append(embed_text(text_content))
                    else:
                        st.error(f"Failed to process {url}: {text_content}")

            if documents:
                st.success("Content from URLs processed successfully!")
                query_embedding = embed_text(question)
                with st.spinner("Searching for relevant information..."):
                    search_results = search_text(query_embedding, np.array(embeddings_list), documents)

                if search_results:
                    st.subheader("Retrieved Context:")
                    for doc, similarity in search_results:
                        st.info(f"Similarity: {similarity:.4f}\n{doc[:500]}...") # Display first 500 chars

                    st.subheader("Answer:")
                    with st.spinner("Generating answer with OpenAI..."):
                        answer = answer_question_openai(question, search_results)
                        st.write(answer)
                else:
                    st.warning("No relevant information found in the processed content.")
            else:
                st.warning("No content was successfully processed from the provided URLs.")
    elif not urls_input:
        st.warning("Please enter one or more URLs.")
    elif not question:
        st.warning("Please enter a question.")

