import pandas as pd
import streamlit as st
import os
import json
import csv
import tiktoken
import numpy as np
import time
import re

#sklearn cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

from config import MODELS, TEMPERATURE, MAX_TOKENS, EMBEDDING_MODELS, PROCESSED_DOCUMENTS_DIR, REPORTS_DOCUMENTS_DIR

import openai

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# USE THIS EMBEDDING FUNCTION THROUGHOUT THIS FILE
# Will download the model the first time it runs
embedding_function = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODELS[0],
    cache_folder="../models/sentencetransformers",
)

# get embedding for one sentence
def get_embedding(sentence):
    try:
        return embedding_function.embed_documents([sentence])[0]
    except Exception as e:
        print(e)
        return np.zeros(384)

def get_retriever(faiss_dir, top_k=5):
    db = FAISS.load_local(faiss_dir, embedding_function)
    retriever = VectorStoreRetriever(vectorstore=db, search_kwargs={"k": top_k})
    return retriever

# make sure load_dotenv is run from main app file first
if os.getenv('OPENAI_API_KEY'):
    openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_BASE'):
    openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_TYPE'):
    openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_VERSION'):
    openai.api_version = os.getenv('OPENAI_API_VERSION')

def initialize_session_state():
    """ Initialise all session state variables with defaults """
    SESSION_DEFAULTS = {
        "cleared_responses" : False,
        "generated_responses" : False,
        "chat_history": [],
        "uploaded_file": None,
        "generation_model": MODELS[0],
        "general_context": "",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": []
    }

    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_response(prompt, model, system_prompt="", temperature=0):
    response = "No model selected"

    if model.startswith("OpenAI: "):
        try:
            response_full = openai.chat.completions.create(model=model[8:], messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt }],temperature=temperature)
        except:
            st.warning("OpenAI API call failed. Waiting 5 seconds and trying again.")
            time.sleep(5)
            response_full = openai.chat.completions.create(model=model[8:], messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt }],temperature=temperature)
    response = response_full.choices[0].message.content

    return response


def create_knowledge_base(docs, faiss_dir="../data/faiss-db/"):
    """Create knowledge base for chatbot."""

    print(f"Loading {PROCESSED_DOCUMENTS_DIR}")
    docs_orig = docs
        
    print(f"Splitting {len(docs_orig)} documents")

    ###### ADD YOUR CODE HERE ######
    ###### ADD YOUR CODE HERE ######
    ###### ADD YOUR CODE HERE ######
    ###### ADD YOUR CODE HERE ######
    ###### ADD YOUR CODE HERE ######
        
    print(f"Created {len(docs)} documents")

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    print("""
        Computing embedding vectors and building FAISS db.
        WARNING: This may take a long time. You may want to increase the number of CPU's in your noteboook.
        """
    )
    db = FAISS.from_texts(texts, embedding_function, metadatas=metadatas)  
    # Save the FAISS db 
    db.save_local(faiss_dir)

    print(f"FAISS VectorDB has {db.index.ntotal} documents")
    

def generate_kb_response(prompt, model, retriever, system_prompt="",template=None, temperature=0):

    if not model.startswith("OpenAI: "):
        return "Please select an OpenAI model."

    relevant_docs = retriever.get_relevant_documents(prompt)

    # string together the relevant documents
    relevant_docs_str = ""
    for doc in relevant_docs:
        relevant_docs_str += doc.page_content + "\n\n"

    if template is None:
        prompt_full = f"""Answer based on the following context

        {relevant_docs_str}

        Question: {prompt}"""
    else:
        prompt_full = template.format(prompt=prompt, context=relevant_docs_str)

    response = generate_response(prompt_full, model=model, system_prompt=system_prompt, temperature=temperature)
    
    return {'answer':response}

