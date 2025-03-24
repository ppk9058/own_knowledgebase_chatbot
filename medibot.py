import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# Add this import for the load_qa_chain function
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import tiktoken
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", convert_system_message_to_human=True)

#App interface
st.title('MediBot')
st.write('Welcome to MediBot! Please enter your symptoms below.')

# File upload and text input
pdf=st.file_uploader('Upload an file ', type=['jpg', 'pdf','png', 'jpeg'])
#st.text_area('Enter your symptoms here', height=200)  
#st.button('Submit')

if pdf is not None:
    st.write(pdf)
    st.write(pdf.name)
    st.write(pdf.type)
    st.write(pdf.size)
    # Process PDF
    pdf_object = PdfReader(pdf)
    text = ''
    for page in pdf_object.pages[:50]:
        text += page.extract_text()
    # Process PDF
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings  # Fixed parameter name
    )
    
    query = st.text_input('Enter your query here')
    if query:
        similar_chunks = vectorstore.similarity_search(query=query, k=2)
        # Create the QA chain here, inside the if query block
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        
        response = chain.run(input_documents=similar_chunks, question=query)

        st.write(response)
        st.write("The Reference Docs for this answer is: ")
        st.write(similar_chunks[0])
        #st.write(similar_chunks[1])
        #st.write(similar_chunks[2])