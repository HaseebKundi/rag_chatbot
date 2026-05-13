import streamlit as st
from rag_pipeline import build_rag_chain

# ================= PAGE CONFIG =================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")

# ================= LOAD PIPELINE =================
@st.cache_resource
def load_chain():
    return build_rag_chain()

rag = load_chain()

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= CHAT HISTORY =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ================= INPUT =================
query = st.chat_input("Ask something...")

if query:

    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        try:
            result = rag(query)

            answer = result["answer"]
            contexts = result["contexts"]

        except Exception as e:
            answer = f"Error: {str(e)}"
            contexts = []

    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # ================= CONTEXT VIEW =================
    if contexts:
        with st.expander("🔎 Retrieved Contexts"):
            for i, doc in enumerate(contexts[:3]):
                st.write(f"Chunk {i+1}")
                st.write(doc.page_content)
                st.write("---")