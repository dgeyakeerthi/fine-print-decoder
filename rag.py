import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub


def extract_section_clause(text):
    section = "Unknown"
    clause = "Unknown"

    section_patterns = [
        r"(Section\s*\d+(\.\d+)*)",
        r"(Sec\.?\s*\d+(\.\d+)*)",
        r"(Q\.?\s*\d+)",
        r"(Chapter\s*\d+)",
        r"(\d+\.\d+)"
    ]

    clause_patterns = [
        r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\b",
        r"\b\d+\)",
        r"\b\d+\.",
        r"\b[a-zA-Z]\)",
        r"\(\w\)"
    ]

    for pattern in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            section = match.group(1)
            break

    for pattern in clause_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            clause = match.group(0)
            break

    return section, clause


def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    current_section = "Unknown"

    for chunk in chunks:
        text = chunk.page_content

        section, clause = extract_section_clause(text)

        if section != "Unknown":
            current_section = section
        else:
            section = current_section

        chunk.metadata["section"] = section
        chunk.metadata["clause"] = clause

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    return db


def ask_question(db, query, chat_history):
    retriever = db.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

  
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

    history_text = "\n".join(
        [f"User: {h['q']}\nAssistant: {h['a']}" for h in chat_history[-3:]]
    )

    prompt = f"""
You are an intelligent assistant.

RULES:
- Answer ONLY using the context
- Be clear and concise
- ALWAYS include Section and Clause
- Format: Answer (Section X, Clause Y)
- If not found, say: Not mentioned in document

Chat History:
{history_text}

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)

    return answer, docs