# 📄 Fine Print Decoder

A simple AI-based application that helps users understand PDF documents by answering questions using **Retrieval-Augmented Generation (RAG)**.

The system retrieves relevant content from the document and provides answers along with **Section and Clause references**.



## 🚀 Features

* Upload any PDF document
* Ask questions in natural language
* Retrieves relevant content using FAISS
* Generates answers using a local LLM (Ollama - LLaMA3)
* Displays **Section, Clause, and snippet** for each answer
* Maintains chat history


## 🧠 Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Ollama (LLaMA 3)



## 📁 Project Structure


fine-print-decoder/
│── app.py          # Streamlit UI
│── rag.py          # RAG logic
│── requirements.txt
│── README.md




## ⚙️ Setup

1. Clone the repository:


git clone <your-repo-link>
cd fine-print-decoder


2. Create virtual environment:


python -m venv venv
venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt


4. Install Ollama and run model:


ollama pull llama3
ollama run llama3


5. Run the app:

streamlit run app.py




