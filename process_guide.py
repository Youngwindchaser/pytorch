import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# --- 1. Load Environment Variables ---
load_dotenv()  # Load environment variables from .env file
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# --- 2. Load the PDF ---
# Using the full path to the PDF in Downloads folder
pdf_path = r"C:\Users\ritik\Downloads\What are the symbols(dragonfly, water, poop, snake.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

if not documents:
    print("Could not load any documents from the PDF. Check the file path and content.")
else:
    print(f"Loaded {len(documents)} document(s) from the PDF.")

    # --- 3. Chunk the Document ---
    # This splits the loaded documents into smaller chunks for better processing.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)
    
    print(f"Split the document into {len(all_splits)} chunks.")

    # --- 4. Create and Store Embeddings in ChromaDB ---
    # This process converts the text chunks into vectors and stores them.
    # The 'persist_directory' is where the database will be saved on your disk.
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    
    print("Successfully created and stored embeddings in the vector database.")