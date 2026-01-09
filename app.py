import os
import streamlit as st
import shutil
from datetime import datetime
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="IndiaSportsHub HR Assistant",
    layout="wide"
)

CONFIDENCE_THRESHOLD = 1.2  # Chroma L2 distance (lower = better)

# --------------------------------------------------
# LOAD API KEY FROM STREAMLIT SECRETS (DEPLOYMENT SAFE)
# --------------------------------------------------
if "GOOGLE_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("GOOGLE_API_KEY not found in Streamlit secrets.")
    st.stop()

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
.category-tag {
    background-color: #e0e0e0;
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 12px;
    color: #333;
    display: inline-block;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=st.session_state.api_key,
        temperature=0.1,
        safety_settings=None
    )

# --------------------------------------------------
# QUERY CATEGORIZATION (HYBRID CONTROL)
# --------------------------------------------------
def get_query_category(query: str):
    q = query.lower()
    if "leave" in q or "vacation" in q or "holiday" in q:
        return "Leave Policy"
    if "benefit" in q or "insurance" in q or "health" in q:
        return "Benefits"
    if "remote" in q or "wfh" in q or "coffee" in q:
        return "Remote Work"
    if "legal" in q or "law" in q or "compliance" in q:
        return "Legal"
    return "General HR"

# --------------------------------------------------
# DOCUMENT INGESTION
# --------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = initialize_embeddings()
    st.session_state.vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_hr"
    )
    st.session_state.vector_store.persist()
    st.session_state.chunk_count = len(chunks)
    st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --------------------------------------------------
# RAG PIPELINE WITH SAFETY CONTROLS
# --------------------------------------------------
def answer_question(question: str):
    embeddings = initialize_embeddings()

    if not os.path.exists("./chroma_db_hr"):
        return (
            "Please upload HR documents first.",
            [],
            "Unknown"
        )

    vector_store = Chroma(
        persist_directory="./chroma_db_hr",
        embedding_function=embeddings
    )

    retrieved = vector_store.similarity_search_with_score(question, k=3)
    top_score = retrieved[0][1]

    category = get_query_category(question)

    # Confidence gate (hallucination prevention)
    if top_score > CONFIDENCE_THRESHOLD:
        return (
            "I'm sorry, the requested information is not available in the uploaded HR documents.",
            [],
            category
        )

    context = "\n\n".join([doc.page_content for doc, _ in retrieved])

    prompt = ChatPromptTemplate.from_template("""
You are an HR Assistant.
Answer strictly using the context below.
If the answer is not clearly present, say you do not have the information.

Context:
{context}

Question:
{question}

Answer:
""")

    llm = initialize_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content, retrieved, category

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main():
    st.header("IndiaSportsHub – AI HR Onboarding Assistant 🤖")

    # ---------------- SIDEBAR (ADMIN) ----------------
    with st.sidebar:
        st.title("Admin Dashboard")
        st.success("API Key loaded securely")

        st.markdown("---")
        st.subheader("Knowledge Base Stats")

        if "chunk_count" in st.session_state:
            st.metric("Indexed Chunks", st.session_state.chunk_count)
            st.caption(f"Last Updated: {st.session_state.last_updated}")
        else:
            st.caption("No documents indexed yet.")

        if st.button("🗑️ Clear Knowledge Base"):
            shutil.rmtree("./chroma_db_hr", ignore_errors=True)
            st.session_state.pop("chunk_count", None)
            st.success("Knowledge base cleared.")
            st.rerun()

        st.markdown("---")
        st.subheader("Upload HR Policy PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process & Index"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Indexing documents..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    create_vector_store(chunks)
                    st.success(f"Indexed {len(chunks)} chunks.")

        st.markdown("---")
        eval_mode = st.checkbox("🧪 Enable Evaluation Mode")
        if eval_mode:
            st.caption("Predefined test queries")
            for q in [
                "What is the leave policy?",
                "Do I get health benefits?",
                "Can I work from a coffee shop?",
                "Who is the CEO?"
            ]:
                if st.button(q):
                    st.session_state.eval_query = q

    # ---------------- CHAT UI ----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "category" in msg:
                st.markdown(
                    f'<span class="category-tag">{msg["category"]}</span>',
                    unsafe_allow_html=True
                )
            st.markdown(msg["content"])

            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 Retrieval Details"):
                    for i, (doc, score) in enumerate(msg["sources"]):
                        st.markdown(f"**Document {i+1}** | Distance: `{score:.4f}`")
                        st.caption(doc.page_content[:200] + "...")

    if "eval_query" in st.session_state:
        user_query = st.session_state.pop("eval_query")
    else:
        user_query = st.chat_input("Ask about HR policies...")

    if user_query:
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources, category = answer_question(user_query)

                st.markdown(
                    f'<span class="category-tag">{category}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(answer)

                if sources:
                    with st.expander("🔍 Retrieval Details"):
                        for i, (doc, score) in enumerate(sources):
                            st.markdown(f"**Document {i+1}** | Distance: `{score:.4f}`")
                            st.caption(doc.page_content[:200] + "...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "category": category,
                    "sources": sources
                })

# --------------------------------------------------
if __name__ == "__main__":
    main()
