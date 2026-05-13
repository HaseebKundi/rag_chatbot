from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os
import pickle

load_dotenv()

# ================= CONFIG =================
INDEX_PATH = "faiss_index"
FILE_PATH = "index.txt"
CHUNK_PATH = "chunks.pkl"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ================= CHUNKING =================
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


# ================= VECTOR STORE =================
def get_vectorstore(chunks):
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_PATH)
    return vs


# ================= RETRIEVER =================
def get_retriever(vs, chunks):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5

    dense = vs.as_retriever(search_kwargs={"k": 5})

    return EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[0.6, 0.4]
    )


# ================= LLM =================
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ================= QUERY REWRITER =================
def rewrite_query(llm, query):
    prompt = PromptTemplate.from_template("""
Convert this into a clean search query:

{question}
""")

    chain = prompt | llm
    result = chain.invoke({"question": query}).content

    return result.strip() if result else query


# ================= FORMAT DOCS =================
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ================= MAIN PIPELINE =================
def build_rag_chain():

    chunks = load_and_chunk()
    vs = get_vectorstore(chunks)
    retriever = get_retriever(vs, chunks)

    llm = get_llm()

    def rag(query):

        clean_query = rewrite_query(llm, query)

        docs = retriever.invoke(clean_query)

        context = format_docs(docs)

        prompt = f"""
        Answer ONLY using context.

        Context:
        {context}

        Question:
        {query}
            """

        answer = llm.invoke(prompt).content

        return {
            "answer": answer,
            "contexts": docs
        }

    return rag
