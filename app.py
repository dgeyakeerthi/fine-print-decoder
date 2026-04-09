import streamlit as st
from rag import process_pdf, ask_question

st.set_page_config(page_title="Fine Print Decoder", layout="wide")

st.title("📄 Fine Print Decoder (AI)")
st.caption("Ask questions from your PDF with source attribution")

# ✅ Session state
if "db" not in st.session_state:
    st.session_state.db = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "file_name" not in st.session_state:
    st.session_state.file_name = None


with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []

    if st.button("🔄 Reset App"):
        st.session_state.chat = []
        st.session_state.db = None
        st.session_state.file_name = None


uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    if st.session_state.file_name != uploaded_file.name:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"✅ Uploaded: {uploaded_file.name}")

        with st.spinner("Processing PDF..."):
            st.session_state.db = process_pdf("temp.pdf")

        st.session_state.file_name = uploaded_file.name
        st.session_state.chat = []  # reset chat on new file


if st.session_state.db:
    query = st.chat_input("Ask something about the document...")

    if query:
        with st.spinner("Thinking... 🤖"):
            answer, docs = ask_question(
                st.session_state.db,
                query,
                st.session_state.chat
            )

            st.session_state.chat.append({
                "q": query,
                "a": answer,
                "docs": docs
            })


for chat in st.session_state.chat:
    with st.chat_message("user"):
        st.write(chat["q"])

    with st.chat_message("assistant"):
        st.write(chat["a"])

        st.markdown("#### 📍 Source Attribution")

        for doc in chat["docs"]:
            section = doc.metadata.get("section", "Unknown")
            clause = doc.metadata.get("clause", "Unknown")
            page = doc.metadata.get("page", "N/A")

            st.markdown(f"""
**Section:** {section}  
**Clause:** {clause}  
**Page:** {page}  

**Snippet:**  
{doc.page_content[:200]}...
""")
            st.write("---")