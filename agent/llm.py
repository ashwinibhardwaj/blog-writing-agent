from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


def get_writer_llm(): 
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")  
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key= api_key
    )

    return llm

def get_generic_llm(): 
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")  
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key= api_key
    )

    return llm