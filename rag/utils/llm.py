from langchain_ollama import ChatOllama

ollama = ChatOllama(model="llama3.3:70b-instruct-q2_K", temperature=0)

def llm():
    # LLM
    return ollama
