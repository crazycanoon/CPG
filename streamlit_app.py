import streamlit as st
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os

# Set your OpenAI API key using Streamlit Secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Function to scrape text content from a URL with enhanced error handling
def scrape_text(url):
    try:
        st.info(f"Attempting to scrape content from: {url}")
        response = requests.get(url, timeout=15)  # Increased timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract main text content (you might need to adjust selectors based on website structure)
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n', strip=True) # More robust text extraction
        if not text.strip():
            return f"Warning: No significant text content found on {url}."
        return text
    except requests.exceptions.Timeout:
        return f"Error: Request to {url} timed out."
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while processing {url}: {e}"

# Function to embed text using OpenAI API with error handling
def embed_text_openai(text):
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        if response and response['data']:
            return response['data'][0]['embedding']
        else:
            return "Error: Received an empty or unexpected response from OpenAI embedding API."
    except openai.error.OpenAIError as e:
        return f"OpenAI API error during embedding: {e}"
    except Exception as e:
        return f"An unexpected error occurred during OpenAI embedding: {e}"

# Function to perform vector search (generally less prone to errors here)
def search_text(query_embedding, embeddings, documents, top_n=3):
    if not embeddings or not documents:
        return []
    if len(embeddings) != len(documents):
        st.error("Error: Mismatch between the number of embeddings and documents.")
        return []
    try:
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        sorted_indices = np.argsort(similarities, axis=0)[::-1]
        results = [(documents[i], similarities[i]) for i in sorted_indices[:top_n]]
        return results
    except Exception as e:
        st.error(f"Error during vector search: {e}")
        return []

# Function to answer the question using OpenAI API with error handling
def answer_question_openai(question, context):
    combined_context = "\n\n".join([doc for doc, _ in context])
    prompt = f"Based on the following information:\n\n{combined_context}\n\nAnswer the question: {question}"
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # Or another suitable model
            prompt=prompt,
            max_tokens=250,  # Adjusted max tokens
            n=1,
            stop=None,
            temperature=0.2,  # Adjust for creativity vs. accuracy
        )
        if response and response.choices:
            return response.choices[0].text.strip()
        else:
            return "Error: Received an empty or unexpected response from OpenAI completion API."
    except openai.error.OpenAIError as e:
        return f"OpenAI API error during completion: {e}"
    except Exception as e:
        return f"An unexpected error occurred during OpenAI completion: {e}"

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
            successful_urls = []
            for url in urls:
                text_content = scrape_text(url)
                if "Error" not in text_content and "Warning" not in text_content:
                    documents.append(text_content)
                    embedding = embed_text_openai(text_content)
                    if isinstance(embedding, str):
                        st.error(f"Error embedding content from {url}: {embedding}")
                    else:
                        embeddings_list.append(embedding)
                        successful_urls.append(url)
                else:
                    st.error(f"Failed or had issues processing {url}: {text_content}")

            if documents and embeddings_list:
                st.success(f"Content from {len(successful_urls)} out of {len(urls)} URLs processed successfully!")
                query_embedding = embed_text_openai(question)
                if isinstance(query_embedding, str):
                    st.error(f"Error embedding your question: {query_embedding}")
                else:
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
                st.warning("No content was successfully processed and embedded from the provided URLs.")
    elif not urls_input:
        st.warning("Please enter one or more URLs.")
    elif not question:
        st.warning("Please enter a question.")

