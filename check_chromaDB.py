import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# ChromaDB configuration
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "rfp_docs"

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Check ChromaDB
try:
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    count = vectorstore._collection.count()
    print(f"ChromaDB collection size: {count}")
except Exception as e:
    print(f"Error connecting to ChromaDB: {str(e)}") 