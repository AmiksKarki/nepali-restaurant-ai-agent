from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


model= OllamaLLM(model="llama3.2")
template= """
You are an expert in answering questions about the Nepali Authentic Cuisine Restaurant.
Please answer the following question based on the provided context.
Here are the reviews of the restaurant:
{reviews}
Here is the question:
{question}

"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n" + "="*50)
    print("\nWelcome to the Nepali Authentic Cuisine Restaurant Q&A!")
    user_input = input("Enter your question (or type 'q' to quit): ")
    print("\n" + "="*50)
    if user_input.lower() == 'q':
        break
    # The retriever will fetch relevant reviews based on the user input from our vector store
    # the retriever will return the top 5 reviews that are most relevant to the user input using similarity search algorithm
    
    reviews_docs = retriever.invoke(user_input)
    
    # Format the reviews for better context
    formatted_reviews = ""
    for i, doc in enumerate(reviews_docs, 1):
        formatted_reviews += f"Review {i}:\n"
        formatted_reviews += f"Content: {doc.page_content}\n"
        formatted_reviews += f"Rating: {doc.metadata.get('rating', 'N/A')}/5\n"
        formatted_reviews += f"Date: {doc.metadata.get('date', 'N/A')}\n"
        formatted_reviews += "-" * 50 + "\n"
    
    if not formatted_reviews:
        formatted_reviews = "No relevant reviews found for your question."
    
    result = chain.invoke({
        "reviews": formatted_reviews,
        "question": user_input
    })
    print(result)
    print("\n" + "="*50)



