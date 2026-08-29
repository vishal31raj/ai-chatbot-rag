# Phase 1 imports
import os
import streamlit as st

# Phase 2 imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Phase 3 imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------

st.title("RAG Chatbot!")

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------

@st.cache_resource
def get_vectorstore():

    pdf_name = "document.pdf"

    # 1. Load PDF
    loader = PyPDFLoader(pdf_name)
    documents = loader.load()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Store embeddings in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# --------------------------------------------------
# LLM
# --------------------------------------------------

groq_chat = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model="openai/gpt-oss-20b"
)


# --------------------------------------------------
# RAG Prompt
# --------------------------------------------------

rag_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the context provided below.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# Chat
# --------------------------------------------------

prompt = st.chat_input("Pass your prompt here!")

if prompt:

    # Display user message
    st.chat_message("user").markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    try:

        # Get vector database
        vectorstore = get_vectorstore()

        if vectorstore is None:
            st.error("Failed to load the document!")
            st.stop()

        # --------------------------------------------------
        # Retrieve relevant documents
        # --------------------------------------------------

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        retrieved_docs = retriever.invoke(prompt)

        # Extract text from retrieved documents
        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        # --------------------------------------------------
        # Create RAG chain
        # --------------------------------------------------

        chain = rag_prompt | groq_chat | StrOutputParser()

        # Send retrieved context + question to LLM
        response = chain.invoke({
            "context": context,
            "question": prompt
        })

        # Display response
        st.chat_message("assistant").markdown(response)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as e:

        st.error(f"Error: [{str(e)}]")

