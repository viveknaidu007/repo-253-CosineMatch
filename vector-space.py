import os
import re
import math
from collections import defaultdict, Counter
import numpy as np
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import wordninja


# in this func the query is undergone the following steps:
#   1. tokenize, 
#   2. converted to lowercase,
#   3. stopwords are removed using stopwords.txt file as well as nltk,
#   4. URLS, punctuations, and filtering
#   5. stemming using porter stemmer

def preprocess_text(text, stopwords_file):
    text = text.lower()
    tokens = word_tokenize(text)
    
    # Filtering tokens by removing URLs, punctuation, and splitting long words
    filtered_tokens = []
    for token in tokens:
        if not token.startswith("http://") and not token.startswith("https://"):
            if re.match("^[^\W\d_]+$", token):
                if len(token) > 10:
                    processed_words = split_words([token])
                    filtered_tokens.extend(processed_words)
                else:
                    filtered_tokens.append(token)
    
    # Remove stopwords
    with open(stopwords_file, 'r') as file:
        custom_stopwords = file.read().splitlines()

    nltk_stopwords = set(stopwords.words('english'))
    all_stopwords = set(custom_stopwords + list(nltk_stopwords))
    filtered_tokens = [token for token in filtered_tokens if token not in all_stopwords]
    
    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    return stemmed_tokens

def split_words(word_list):
    new_word_list = []
    for word in word_list:
        if len(word) > 10:
            split_words = wordninja.split(word)
            new_word_list.extend(split_words)
        else:
            new_word_list.append(word)
    return new_word_list

#""" making inverted index by receiving a collection of txt files reading them in a loop one by one
#    and sending the txt file to the preprocess function for preprocessing 
#    and finally add the tokens in the inverted index
#"""
def inverted_index(directories, stopwords_file):
    inverted_index = defaultdict(set)
    for directory in directories:
        for filename in os.listdir(directory):
            try:
                with open(os.path.join(directory, filename), 'r') as file:
                    text = file.read()
                    tokens = preprocess_text(text, stopwords_file)
                    for token in tokens:
                        inverted_index[token].add(filename)
            except UnicodeDecodeError:
                print(f"Error reading file: {filename}")

    return inverted_index

#""" making term frequency dic by receiving inverted index"""
def term_frequency(index):
    term_freq = {}
    for term, documents in index.items():
        term_freq[term] = Counter(documents)
        total_tokens = sum(term_freq[term].values())
        term_freq[term] = {doc: freq / total_tokens for doc, freq in term_freq[term].items()}

    return term_freq

#""" making doc frequency dic by receiving inverted index"""
def document_frequency(index):
    doc_freq = {}
    for term, documents in index.items():
        doc_freq[term] = len(documents)
    return doc_freq

#"""Calculate TF-IDF weights for each term in the document corpus.
#TF-IDF (Term Frequency-Inverse Document Frequency) 
#used to evaluate the importance of a term in a document relative to a corpus."""
def tfidf(tf, df, num_docs):
    tfidf_weights = {}
    for term, doc_freq in tf.items():
        tfidf_weights[term] = {}
        for doc, term_freq in doc_freq.items():
            tfidf_weights[term][doc] = term_freq * math.log(num_docs / df[term])
    return tfidf_weights

#"""Calculate TF-IDF weights for each term in the query.
#This function calculates the TF-IDF (Term Frequency-Inverse Document Frequency) 
#weights for each term in the query, which are used to rank documents 
#based on their relevance to the query."""
def calculate_tfidf_query(query, preprocessed_query, doc_freq, total_docs):
    query_tf = Counter(preprocessed_query)
    total_query_tokens = sum(query_tf.values())
    query_tf = {term: freq / total_query_tokens for term, freq in query_tf.items()}
    
    query_idf = {}
    for term in preprocessed_query:
        if term in doc_freq:
            query_idf[term] = math.log(total_docs / doc_freq[term])
        else:
            query_idf[term] = 0  # Term not present in any document
    
    query_tfidf_weights = {}
    for term, tf in query_tf.items():
        query_tfidf_weights[term] = tf * query_idf[term]

    return query_tfidf_weights

#"""Vectorize the query using TF-IDF weights and normalize the resulting vector.
#This function converts the query TF-IDF weights into a vector representation 
#and normalizes it to ensure consistent scaling.
#"""
def vectorize_query(query_tfidf_weights, vocabulary):
    query_vector = np.zeros(len(vocabulary))
    for i, term in enumerate(vocabulary):
        if term in query_tfidf_weights:
            query_vector[i] = query_tfidf_weights[term]
    query_magnitude = np.linalg.norm(query_vector)
    if query_magnitude != 0:
        query_vector /= query_magnitude

    return query_vector

#"""Vectorize the documents using TF-IDF weights and normalize the resulting vectors.
#This function converts the TF-IDF weights of documents into vector representations 
#and normalizes each document vector to ensure consistent scaling.
#"""
def vectorize_documents(document_ids, vocabulary, tfidf_weights):
    num_documents = len(document_ids)
    num_terms = len(vocabulary)
    document_vectors = np.zeros((num_documents, num_terms))

    for i, doc_id in enumerate(document_ids):
        for j, term in enumerate(vocabulary):
            if doc_id in tfidf_weights[term]:
                document_vectors[i, j] = tfidf_weights[term][doc_id]

    document_magnitudes = np.linalg.norm(document_vectors, axis=1)
    zero_magnitudes = np.where(document_magnitudes == 0)[0]
    document_magnitudes[zero_magnitudes] = 1
    document_vectors /= document_magnitudes[:, np.newaxis]

    return document_vectors

#"""Compute the cosine similarity between a query vector and a document vector.
#This function calculates the cosine similarity between the query and document vectors,
#which are both normalized representations of TF-IDF weights. It computes the dot 
#product of the two vectors and divides it by the product of their magnitudes.
#If either of the magnitudes is zero, it returns 0 to avoid division by zero.
#"""
def cosine_similarity(query_vector, document_vector):
    dot_product = np.dot(query_vector, document_vector)
    query_magnitude = np.linalg.norm(query_vector)
    doc_magnitude = np.linalg.norm(document_vector)
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
    else:
        return dot_product / (query_magnitude * doc_magnitude)


def rank_documents(similarity_scores, threshold, k=5):
    sorted_similarities = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    top_documents = [doc_id for doc_id, score in sorted_similarities if score > threshold][:k]
    return top_documents

def search_documents():
    query = st.text_input("Enter your Query:")
    if st.button("Search"):
        preprocessed_query = preprocess_text(query, stopwords_file)
        query_tfidf_weights = calculate_tfidf_query(query, preprocessed_query, doc_freq, total_docs)
        query_vector = vectorize_query(query_tfidf_weights, vocabulary)

        similarities = {}
        for i, doc_vector in enumerate(document_vectors):
            similarity = cosine_similarity(query_vector, doc_vector)
            similarities[document_ids[i]] = similarity

        threshold = 0.0
        top_documents = rank_documents(similarities, threshold)
        
        st.subheader("Top ranked documents:")
        for doc_id in top_documents:
            st.write(doc_id)

# Main function (entry point for Streamlit)
def main():
    global stopwords_file, doc_freq, total_docs, vocabulary, document_ids, document_vectors

    # Initialize directories and stopwords_file
    directories = [r"C:\Users\user\Documents\6th semester\IR\ResearchPapers"]
    stopwords_file = r"C:\Users\user\Documents\6th semester\IR\Stopword-List.txt"

    # Create inverted index and calculate TF-IDF weights
    index = inverted_index(directories, stopwords_file)
    term_freq = term_frequency(index)
    doc_freq = document_frequency(index)
    total_docs = len(os.listdir(directories[0]))
    tfidf_weights = tfidf(term_freq, doc_freq, total_docs)

    # Get vocabulary
    vocabulary = list(tfidf_weights.keys())

    # Vectorize documents
    document_ids = os.listdir(directories[0])
    document_vectors = vectorize_documents(document_ids, vocabulary, tfidf_weights)

    # Create Streamlit app
    st.title("Vector Space Model")

    # Sidebar for research papers
    st.sidebar.title("Research Papers")
    selected_paper = st.sidebar.selectbox("Select a paper", document_ids)
    if st.sidebar.button("Read Paper"):
        successful_read = False
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1']
        for encoding in encodings_to_try:
            try:
                with open(os.path.join(directories[0], selected_paper), 'r', encoding=encoding) as file:
                    paper_content = file.read()
                    st.subheader(selected_paper)
                    st.write(paper_content)
                    successful_read = True
                    break
            except UnicodeDecodeError:
                continue
        
        if not successful_read:
            st.error(f"Error: Unable to read {selected_paper}. File encoding may not be recognized.")
        
    search_documents()

if __name__ == "__main__":
    main()
