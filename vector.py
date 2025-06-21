# importing all the necessary libraries
# to work with the vector store and embeddings
# OllamaEmbeddings for generating embeddings
# Chroma for managing the vector store
# Document for creating documents with metadata
# pandas for reading the CSV file containing reviews
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the restaurant reviews from a CSV file
df = pd.read_csv("reviews.csv")

# Define the embedding model from ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define the location for the Chroma vector store
db_location= "./chroma_db"

# Create a Chroma vector store with the specified collection name and embedding function
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory= db_location,
    embedding_function=embeddings
    )

# Check if the collection actually has documents, not just if directory exists
collection_count = vector_store._collection.count()
add_documents = collection_count == 0

if add_documents:
    print("No documents found in vector store. Adding documents...")
    documents=[]
    ids=[]
    for i, row in df.iterrows():
        document= Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={
                "rating": row["Rating"],
                "date": row["Date"]
            },
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
    
    # Add the documents to the vector store
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Added {len(documents)} documents to the vector store.")
else:
    print(f"Vector store already contains {collection_count} documents. Skipping document addition.")

# creating a retriever from the vector store that will be used to retrieve relevant documents based on user queries
retriever = vector_store.as_retriever(
    search_kwargs={ "k":5 }
)