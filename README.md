# 🤖 AI-Powered HR Onboarding Assistant (RAG-Based)

An **AI-powered Self-Service HR Knowledge Assistant** that allows employees to query company HR policies using natural language, while ensuring **accuracy, transparency, and zero hallucinations**.

This project was built as part of an **AI Engineer interview assignment** and focuses on **Retrieval-Augmented Generation (RAG)**, safety-first design, and admin usability.

---

## 🚀 Live Demo

**[🚀 Live Demo: https://ai-powered-hr-app-assistant.streamlit.app/](https://ai-powered-hr-app-assistant.streamlit.app/)**

Try the application instantly in your browser!

---

## ✨ Key Features

### 🔍 Intelligent HR Q&A (RAG)
- Employees can ask questions about **leave, benefits, legal policies, and remote work**
- Answers are generated **strictly from uploaded HR documents**
- Uses semantic search + LLM synthesis

### 🛑 Hallucination Prevention (Safety First)
- Confidence threshold on vector similarity
- If relevant information is not found, the assistant **refuses to answer**
- Critical for HR & legal domains

### 📌 Source Transparency & Citations
- Every answer shows:
  - Retrieved document chunks
  - Similarity distance scores
  - Source text snippets
- Makes responses **auditable and trustworthy**

### 🏷️ Hybrid Query Categorization
- Rule-based categorization for control logic
- Categories: Leave Policy, Benefits, Legal, Remote Work, General HR
- Avoids over-reliance on LLM reasoning

### 🧑‍💼 Admin Dashboard
- Upload & index HR policy PDFs
- View knowledge base stats (chunk count, last updated)
- Clear & re-index the entire knowledge base
- API key management

### 🧪 Evaluation Mode
- Built-in test questions for validation
- Includes **out-of-scope queries** to verify refusal behavior
- Helps evaluate retrieval quality and system reliability

---

## 🏗️ System Architecture

```
HR Policy PDFs
↓
Text Extraction (PDF)
↓
Intelligent Chunking
↓
Embedding Generation (MiniLM)
↓
Vector Storage (ChromaDB)
↓
Similarity Search (with Threshold)
↓
LLM (Gemini Flash)
↓
Answer + Category + Sources
```

---

## 🧠 Design Decisions

### Why Retrieval-Augmented Generation (RAG)?
- HR systems require **factual correctness**
- Prevents hallucinations by grounding responses in documents

### Why HuggingFace MiniLM Embeddings?
- Lightweight and fast
- Works locally (no embedding API cost)
- High-quality semantic similarity

### Why ChromaDB?
- Simple local vector database
- Persistent storage
- Ideal for prototypes and interview assignments

### Why Confidence Threshold?
- HR & legal domains are **high-risk**
- It’s safer to say *“I don’t know”* than give a wrong answer

---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|------------|
| Frontend | Streamlit |
| LLM | Google Gemini Flash |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| PDF Parsing | PyPDF |
| Framework | LangChain |
| Language | Python |

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/saoud30/AI-Powered-HR-Onboarding-Assistant.git
cd AI-Powered-HR-Onboarding-Assistant
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

### 4️⃣ Usage Flow

- Enter your Google API Key in the Admin Dashboard
- Upload HR policy PDFs
- Click Process & Index
- Ask questions in the chat interface

## 🧪 Sample Test Questions

- How many vacation days do I get?
- What health benefits are provided?
- Can I work from a coffee shop?
- Who is the CEO? (Expected: refusal)

## ⚠️ Known Limitations

- No user authentication (single-user local app)
- Local vector database only
- PDF-only ingestion (no DOCX/TXT yet)
- No automated evaluation metrics

## 🔮 Future Improvements

- Role-based access (Admin vs Employee)
- Support for DOCX and TXT documents
- Hybrid search (BM25 + Vector)
- RAG evaluation metrics (precision/recall)
- Cloud vector database (Pinecone / Weaviate)
- API-based backend (FastAPI)

## 🏁 Final Notes

This project prioritizes:

- **Correctness** over creativity
- **Transparency** over black-box answers
- **Safety** over convenience

It is production-aligned (not production-hardened) and demonstrates strong fundamentals in building trustworthy AI systems.

---


