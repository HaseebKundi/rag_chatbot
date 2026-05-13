from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os
import pickle

load_dotenv()

INDEX_PATH = "faiss_index"
FILE_PATH = "index.txt"
CHUNK_PATH = "chunks.pkl"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def load_and_chunk():
    if os.path.exists(CHUNK_PATH):
        with open(CHUNK_PATH, "rb") as f:
            return pickle.load(f)
    
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    docs = loader.load()
    chunker = SemanticChunker(embeddings)
    chunks = chunker.split_documents(docs)
    
    with open(CHUNK_PATH, "wb") as f:
        pickle.dump(chunks, f)
    return chunks

def get_vectorstore(chunks):
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_PATH)
    return vs

def get_retriever(vs):
    # Simplified - using only FAISS retriever
    return vs.as_retriever(search_kwargs={"k": 5})

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_rag_chain():
    chunks = load_and_chunk()
    vs = get_vectorstore(chunks)
    retriever = get_retriever(vs)
    llm = get_llm()
    
    def rag(query):
        docs = retriever.invoke(query)
        context = format_docs(docs)
        prompt = f"""Answer ONLY using the context below.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = llm.invoke(prompt).content
        return {"answer": answer, "contexts": docs}
    
    return rag
