import os
import json
from datetime import datetime
import hashlib
import tempfile
from pathlib import Path
import uuid
import time
import streamlit as st
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

# Set page config at the top
st.set_page_config(page_title="Smart Business Guide")

# Set OpenAI API key
try:
    from dotenv import load_dotenv
    load_dotenv('./.env')
except:
    pass

# Configuration for OpenAI's embedding model
class Config:
    MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME = "text-embedding-3-small"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Check if the API key is set
if not Config.OPENAI_API_KEY:
    st.error("This application requires OpenAI API key which is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Set up logging
logging.basicConfig(level=logging.ERROR, filename='app_errors.log')

# Function to save chats to a JSON file
def save_chats():
    temp_filename = "chats_temp.json"
    with open(temp_filename, "w") as f:
        json.dump(st.session_state.chats, f)
    os.replace(temp_filename, "chats.json")

# Function to load chats from a JSON file
def load_chats():
    if os.path.exists("chats.json"):
        try:
            with open("chats.json", "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error("The chats.json file is corrupted and cannot be read. A new file will be created.")
            # Generate a unique filename using timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            corrupted_filename = f"chats_corrupted_{timestamp}.json"
            os.rename("chats.json", corrupted_filename)
            return {}
    return {}

#Function to process PDF files
def process_pdf(file, chunk_size, chunk_overlap):
    file_hash = hashlib.md5(file.getvalue()).hexdigest()
    filename = f"temp_{file_hash}.pdf"
    
    with open(filename, "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()

    file_summary = {'pages':len(pages),'characters':sum([len(x.page_content) for x in pages])}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(pages)

    os.remove(filename)

    return chunks, file.name,file_summary

def create_context(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

def generate_chat_title(context, question):
    llm = ChatOpenAI(
        model_name=Config.MODEL,
        temperature=0,
        streaming=True,
        openai_api_key=Config.OPENAI_API_KEY
    )
    messages = [
        {"role": "system", "content": "Generate a concise, descriptive title for a chat based on the given context and question."},
        {"role": "user", "content": f"Context: {context[:500]}...\n\nQuestion: {question}\n\nTitle:"}
    ]
    response = llm(messages)
    return response.content.strip()

# Function to load OpenAI embedding model
def load_embedding_model(model_name, normalize_embedding=True):
    print("Loading OpenAI embedding model...")
    openai_embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    return openai_embeddings

#Function to create OpenAI embeddings
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    print(f"Creating embeddings ({len(chunks)} chunks)...")
    if not chunks:
        print("Warning: No chunks to process. The PDF might be empty or unreadable.")
        return None
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print('saving vectorstore...',end='')
    vectorstore.save_local(storing_path)
    print(' done')
    return vectorstore

#Load the chain
def load_qa_chain(retriever, llm, prompt):
    print("Loading QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def get_response(query, chain):
    response = chain({'query': query})
    return response['result'].strip()

def switch_to_ChatGPT(prompt, selected_model):
    llm = ChatOpenAI(
        model_name=selected_model,
        temperature=0,
        streaming=True,
        openai_api_key=Config.OPENAI_API_KEY
    )
    messages = [{"role": "user", "content": prompt}]
    # response = llm(messages)

    response = ""
    message_placeholder = st.empty()  # An empty placeholder to update the response in real-time

    for chunk in llm.stream(messages):
        response += chunk.content  # Extract and accumulate the content
        message_placeholder.markdown(response)  # Update the placeholder with Markdown for better formatting
    return response


class PDFHelper:
    def __init__(self, model_name=Config.MODEL):
        self._model_name = model_name
        self.vectorstore = None  # To hold the loaded vector store

    def load_vectorstore(self, vectorstore_path):
        """Load the vector store only once and reuse it."""
        if self.vectorstore is None:
            print("Loading vectorstore from disk...")
            self.vectorstore = FAISS.load_local(
                vectorstore_path, 
                embeddings=load_embedding_model(Config.EMBEDDING_MODEL_NAME),
                allow_dangerous_deserialization=True
            )
        return self.vectorstore

    
    def ask(self, vectorstore_path, question):
        # Load the vectorstore if not already loaded
        vectorstore = self.load_vectorstore(vectorstore_path)

        # Convert the query to embeddings and retrieve relevant chunks
        print('...retrieving relevant context from vectorstore ', end='')
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5,'fetch_k':30,'filter': None}, search_type="mmr")
        retrieved_docs = retriever.get_relevant_documents(question)  # Get relevant chunks
        print(f'... done ({len(retrieved_docs)} documents/chunks)')

        # Extract the content from the retrieved chunks and combine them into context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Define the prompt template
        prompt_template = """
        # Instructions #
        You are an honest and smart assistant.
        You must provide a precise answer to a query from the given context/information.
        If you don't know the answer, say you don't know. Don't make up an answer.
        Strictly follow these rules: 
        i) Analyze the query to provide only the precise answer. Do not provide any additional information unless necessary and helpful in the query's context.
        ii) You do not need to provide the answer exactly in the same words as in the context/information. You may style the answer in appropriate format and words. For example:
        query_example = "query: What are the working hours in Finland?"
        words_in_context = "weekly working time should not exceed 40 hours"
        your_answer = "The working hours in Finland are 40 hours per week."
        iii) Consider creating bullet points or bold headings/sub-headings or other formatting as necessary.

        # Context #
        {context}

        # Question #
        {question}

        # Response #
        Your response:
        """

        # Create a valid PromptTemplate object
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # Set up the OpenAI Chat model with streaming enabled
        llm = ChatOpenAI(
            model_name=self._model_name,
            temperature=0,
            streaming=True,  # Enable streaming
            openai_api_key=Config.OPENAI_API_KEY
        )

        # Initialize the RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}  # Use the PromptTemplate object
        )

        # Create a placeholder for the real-time streaming
        message_placeholder = st.empty()  
        full_response = ""

        # Stream the response from the OpenAI model
        for chunk in chain.stream({"query": question}):
            full_response += chunk['result']  # Accumulate the result
            message_placeholder.markdown(full_response)  # Update the display in real-time

        # Save the retrieved context so it can be displayed
        st.session_state.context = context  # Update the session state with the context

        return full_response.strip(), context  # Return the result and the context


def main():
    # Load chats
    if "chats" not in st.session_state:
        chats = load_chats()
        if chats is None:
            st.error("The chats.json file is corrupted and cannot be read. A new file will be created.")
            st.session_state.chats = {}
        else:
            st.session_state.chats = chats

    st.title("Smart Guide - Enterpreneurship in Finland")

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = ""
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "vectorstore_path" not in st.session_state:
        st.session_state.vectorstore_path = None

    with st.sidebar:
        st.write('This smart guide answers questions from a PDF guide.')
        available_models = ["gpt-4o-mini", "gpt-4o"]
        selected_model = st.selectbox("Select a model", available_models)

        uploaded_file = st.file_uploader("Upload a PDF file (optional)", type="pdf")

        if uploaded_file is not None:
            st.write("PDF mode: Ask questions about the uploaded document.")
            chunk_size = 600
            chunk_overlap = 100

            if st.session_state.current_file != uploaded_file.name:
                chunks, filename, file_summary = process_pdf(uploaded_file, chunk_size, chunk_overlap)
                context = f'Total {len(chunks)} chunks with properties {str(file_summary)}'
                st.session_state.context = context
                st.session_state.current_file = filename

                # Create and save the vectorstore
                vector_store_directory = os.path.join(str(Path.home()), 'langchain-store', 'vectorstore',
                                                      'pdf-doc-helper-store', str(uuid.uuid4()))
                print(f'...saving vectorstore file in {vector_store_directory}')
                os.makedirs(vector_store_directory, exist_ok=True)
                embed = load_embedding_model(model_name=Config.EMBEDDING_MODEL_NAME)
                vectorstore = create_embeddings(chunks=chunks, embedding_model=embed, storing_path=vector_store_directory)
                vectorstore_path = vector_store_directory  # Save the path

                # Store the vectorstore path in session state
                st.session_state.vectorstore_path = vectorstore_path

                chat_title = f"Chat about {filename}"
                st.session_state.current_chat = chat_title
                st.session_state.chats[chat_title] = {
                    'messages': [],
                    'context': context,
                    'file': filename,
                    'vectorstore_path': vectorstore_path  # Store the path here
                }
                st.session_state.messages = []

                system_msg = f"New file uploaded: {filename}. You can now ask questions about this document."
                st.session_state.messages.append({"role": "system", "content": system_msg, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

                save_chats()
                st.rerun()
        else:
            st.write("Regular chat mode: Ask any questions.")

    # Chat selection
    chat_names = ["New Chat"] + list(st.session_state.chats.keys())
    current_chat = st.selectbox("Select a chat", chat_names, index=chat_names.index(st.session_state.current_chat))

    if current_chat != st.session_state.current_chat:
        if current_chat == "New Chat":
            st.session_state.current_chat = "New Chat"
            st.session_state.messages = []
            st.session_state.context = ""
            st.session_state.current_file = None
            st.session_state.vectorstore_path = None  # Reset the vectorstore path
        else:
            st.session_state.current_chat = current_chat
            st.session_state.messages = st.session_state.chats[current_chat]['messages']
            st.session_state.context = st.session_state.chats[current_chat]['context']
            st.session_state.current_file = st.session_state.chats[current_chat]['file']
            # Retrieve the vectorstore path
            st.session_state.vectorstore_path = st.session_state.chats[current_chat].get('vectorstore_path')
        
        st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['timestamp']}**")
            st.markdown(message["content"])

    # Chat input and response handling
    if prompt := st.chat_input("What is your question?"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
        
        with st.chat_message("user"):
            st.markdown(f"**{timestamp}**")
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if st.session_state.current_file is None:
                full_response = switch_to_ChatGPT(prompt, selected_model)
            else:
                pdf_helper = PDFHelper(
                    model_name=selected_model
                )
                vectorstore_path = st.session_state.vectorstore_path
                if vectorstore_path and os.path.exists(vectorstore_path):
                    full_response, context = pdf_helper.ask(
                        vectorstore_path=vectorstore_path,
                        question=prompt
                    )
                else:
                    full_response = "The vectorstore for this chat is missing. Please re-upload the PDF file."
            
            message_placeholder.markdown(full_response)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": timestamp})

        # Update the chat in st.session_state.chats
        if st.session_state.current_chat != "New Chat":
            st.session_state.chats[st.session_state.current_chat]['messages'] = st.session_state.messages

        save_chats()
        st.rerun()

    if st.session_state.current_file:
        st.sidebar.write(f"Current file: {st.session_state.current_file}")
    if st.session_state.context:
        with st.expander("Current Context"):
            st.write(st.session_state.context)

    if st.sidebar.button('New Chat'):
        st.session_state.current_chat = "New Chat"
        st.session_state.messages = []
        st.session_state.context = ""
        st.session_state.current_file = None
        st.session_state.vectorstore_path = None  # Reset the vectorstore path
        st.rerun()

if __name__ == "__main__":
    main()

