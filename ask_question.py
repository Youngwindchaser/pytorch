import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. Load Environment Variables ---
load_dotenv()  # Load environment variables from .env file
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# --- 2. Load the Existing Vector Database ---
# We are now loading the "brain" we created in the previous step.
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# --- 3. Create a Prompt Template ---
# This template instructs the LLM on how to use the context from our guide.
template = """
You are an expert assistant. Your task is to answer the user's question based ONLY on the following context.
If the answer is not available in the context, respond with 'I am sorry, the guide does not contain information on this topic.'
Do not use any prior knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# --- 4. Define the LLM and the RAG Chain ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# This is the RAG chain that connects everything.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. Ask a question! ---
question = "What is the process for updating my profile?" # <-- CHANGE THIS TO YOUR QUESTION
response = rag_chain.invoke(question)

print("Question:", question)
print("Answer:", response)

# --- Example of another question ---
question_2 = "What are the security policies?" # <-- CHANGE THIS
response_2 = rag_chain.invoke(question_2)
print("\nQuestion:", question_2)
print("Answer:", response_2)