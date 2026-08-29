# AI Chatbot RAG

A Retrieval-Augmented Generation (RAG) chatbot built with **Python, Streamlit, LangChain, HuggingFace Embeddings, ChromaDB, and Groq**.

The application loads a PDF document, splits it into smaller chunks, generates embeddings, stores them in a vector database, retrieves relevant information based on the user's question, and uses an LLM to generate the final response.

## Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **HuggingFace Embeddings**
* **ChromaDB**
* **PyPDF**
* **Groq**

## RAG Architecture

```text
PDF Document
     ↓
PyPDFLoader
     ↓
Document Text
     ↓
Text Chunking
     ↓
HuggingFace Embeddings
     ↓
ChromaDB
     ↓
Similarity Search
     ↓
Relevant Context
     ↓
Groq LLM
     ↓
Generated Answer
```

## Project Structure

```text
ai-chatbot-rag/
│
├── server.py
├── document.pdf
├── Pipfile
├── Pipfile.lock
├── .env
├── .env.example
├── .gitignore
└── README.md
```

> **Note:** Do not commit `.env` or private/copyrighted PDF documents to GitHub.

## Prerequisites

Make sure you have installed:

* Python
* Pipenv
* A Groq API key

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ai-chatbot-rag
```

### 2. Create and activate the Pipenv environment

```bash
pipenv shell
```

### 3. Install dependencies

```bash
pipenv install
```

If you don't have a `Pipfile` yet, you can install the required packages with:

```bash
pipenv install streamlit langchain langchain-groq langchain-community langchain-huggingface langchain-text-splitters pypdf chromadb sentence-transformers python-dotenv
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your actual API key to GitHub.

### 5. Add your PDF

Place the PDF you want to query in the project root and name it:

```text
document.pdf
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run server.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open it in your browser and start asking questions about the PDF.

## How It Works

### 1. Load the PDF

`PyPDFLoader` extracts text from the PDF document.

### 2. Split the document

The extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

```text
Chunk Size: 1000
Chunk Overlap: 100
```

### 3. Generate embeddings

HuggingFace's embedding model converts each text chunk into a numerical vector.

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

### 4. Store embeddings

The embeddings are stored in **ChromaDB**, which acts as the vector database.

### 5. Retrieve relevant information

When a user asks a question, the application searches ChromaDB and retrieves the most relevant document chunks.

### 6. Generate the answer

The retrieved context is provided to the Groq-hosted LLM, which generates the final answer based on the document.

## Environment Variables

| Variable       | Description                         |
| -------------- | ----------------------------------- |
| `GROQ_API_KEY` | API key used to access the Groq LLM |

## Future Improvements

* Support multiple PDF documents
* Add document upload through the Streamlit UI
* Persist ChromaDB between application restarts
* Display source documents and page numbers
* Add conversation memory
* Add streaming responses
* Add support for multiple LLM providers
* Implement agentic AI capabilities
* Add tool calling and web search
* Add evaluation and RAG quality metrics

## License

This project is intended for educational and learning purposes.
