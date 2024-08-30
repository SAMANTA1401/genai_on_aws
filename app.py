import json
import os
import sys
import boto3
import streamlit as st 
## we will be using Titan Embedding technique Model for vector embedding
from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader 

# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA


# Bedrock Clients
bedrock = boto3.client(service_name = "bedrock-runtime", region_name = 'us-east-1', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock) #us-east-1

## Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# vector embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )

    vectorstore_faiss.save_local("faiss_index")

    return vectorstore_faiss

# def get_claude_llm():
#     ## create the Anthropic model
#     llm = Bedrock(model_id="", client=bedrock, model_kwargs={'maxTokes':512})
#     return llm

def get_llama2_llm():
    bedrock2 = boto3.client(service_name = "bedrock-runtime", region_name = 'ap-south-1', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    ## create the Anthropic model
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock2, model_kwargs={'max_gen_len':512}) #ap-south-1
    return llm

prompt_template = """
Human: Use the following  pieces of context to provide a concise answer to the question at the end but uses atleast
summarize with 250 words with detailed explanations. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}


Assistant:
"""

PROMPT = PromptTemplate(
    template = prompt_template, input_variables = ['context','question']
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity", search_kwargs={"k":3}
        ),
        return_source_documents = True,
        chain_type_kwargs = {"prompt":PROMPT}
            
        )
    
## go through langchain documentation
    answer = qa({"query":query})
    return answer['result']


## Main Execution
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a Question from the PDF Files")
    with st.sidebar:
        st.title("Menu:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done") 

        # if st.button("cloude Output"):
        #     with st.spinner("Processing..."):
        #         faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings)
        #         llm = get_claude_llm()

        #         #faiss_index = get_vector_store(docs)
        #         st.write(get_response_llm(llm,faiss_index,user_question))
        #         st.success("Done")

    if st.button("llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm = get_llama2_llm()

            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()