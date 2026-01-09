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
from langchain_core.messages import HumanMessage, SystemMessage

# --- CONFIGURATION ---
st.set_page_config(page_title="IndiaSportsHub HR Assistant", layout="wide")

# --- CONSTANTS ---
# Confidence Threshold: Chroma uses L2 distance. 0 is perfect match. 
# Higher than 1.2 usually means weak match.
CONFIDENCE_THRESHOLD = 1.2 

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header {font-size: 30px; font-weight: bold; color: #1f77b4;}
    .category-tag {background-color: #e0e0e0; padding: 4px 10px; border-radius: 10px; font-size: 12px; color: #333; margin-bottom: 10px; display: inline-block;}
    .source-box {background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 4px solid #1f77b4;}
    .score-badge {font-size: 10px; color: #666; font-style: italic;}
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def initialize_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest", # Using user's preference
        google_api_key=api_key,
        temperature=0.1,
        safety_settings=None
    )

# --- FEATURE 2: HYBRID QUERY CATEGORIZATION ---
def get_query_category(query):
    """Rule-based classification for control, before LLM reasoning."""
    q = query.lower()
    if "leave" in q or "vacation" in q or "holiday" in q:
        return "Leave Policy"
    elif "benefit" in q or "insurance" in q or "health" in q:
        return "Benefits"
    elif "legal" in q or "law" in q or "compliance" in q:
        return "Legal"
    elif "remote" in q or "wfh" in q or "coffee" in q:
        return "Remote Work"
    return "General HR"

# --- DATA PROCESSING ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = initialize_embeddings()
    
    # Create/Update Vector Store
    st.session_state.vector_store = Chroma.from_texts(
        texts=text_chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db_hr"
    )
    st.session_state.vector_store.persist()
    
    # --- FEATURE 4: ADMIN STATS ---
    st.session_state.chunk_count = len(text_chunks)
    st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- RAG LOGIC (WITH THRESHOLD & TRANSPARENCY) ---
def user_input(user_question):
    embeddings = initialize_embeddings()
    
    # Load DB
    if os.path.exists("./chroma_db_hr"):
        st.session_state.vector_store = Chroma(
            persist_directory="./chroma_db_hr", 
            embedding_function=embeddings
        )
    else:
        return "Please upload HR documents in the Admin Dashboard first.", [], "Unknown"

    # --- FEATURE 1 & 3: TRANSPARENCY & THRESHOLD ---
    # similarity_search_with_score returns (Document, distance_score)
    # Lower distance = better match.
    retrieved_docs = st.session_state.vector_store.similarity_search_with_score(
        user_question, k=3
    )

    # Check Confidence Threshold (Feature 3)
    # If the top result is too far away (distance > threshold), we refuse to answer.
    top_score = retrieved_docs[0][1]
    
    if top_score > CONFIDENCE_THRESHOLD:
        return (
            "I'm sorry, this information is not available in the HR documents (Confidence Score too low).", 
            [], 
            get_query_category(user_question)
        )

    # Prepare context manually to pass to LLM
    context_text = "\n\n".join([doc.page_content for doc, score in retrieved_docs])
    
    # Generate Answer (Manual LLM call to have full control over context)
    llm = initialize_llm(st.session_state.api_key)
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful HR Assistant for IndiaSportsHub. 
    Answer strictly based on the context below.
    
    Context:
    {context}

    Question: {input}

    Answer:
    """)
    
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "input": user_question})
    
    return response.content, retrieved_docs, get_query_category(user_question)

# --- MAIN APP ---
def main():
    st.header("IndiaSportsHub - AI HR Onboarding Assistant 🤖")

    # SIDEBAR
    with st.sidebar:
        st.title("Admin Dashboard")
        
        # --- ADMIN FEATURE: API KEY ---
        api_key = st.text_input("Google API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
            st.success("API Key saved")
        
        st.markdown("---")

        # --- ADMIN FEATURE: STATS ---
        st.subheader("Knowledge Base Stats")
        if 'chunk_count' in st.session_state:
            st.metric("Indexed Chunks", st.session_state.chunk_count)
            st.caption(f"Last Updated: {st.session_state.last_updated}")
        else:
            st.caption("No data indexed yet.")

        # --- ADMIN FEATURE: DELETE DB ---
        if st.button("🗑️ Clear Knowledge Base"):
            if os.path.exists("./chroma_db_hr"):
                shutil.rmtree("./chroma_db_hr")
                if 'chunk_count' in st.session_state: del st.session_state.chunk_count
                st.success("Database cleared.")
                st.rerun()
        
        st.markdown("---")
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        if st.button("Process & Index"):
            with st.spinner("Processing..."):
                if pdf_docs and api_key:
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success(f"Indexed {len(chunks)} chunks.")
                else:
                    st.error("Please upload PDFs and enter API Key.")

        # --- FEATURE 5: EVALUATION MODE ---
        st.markdown("---")
        eval_mode = st.checkbox("🧪 Enable Evaluation Mode")
        if eval_mode:
            st.info("Running predefined test cases to validate retrieval.")
            test_queries = [
                "What is the leave policy?",
                "Do I get health benefits?",
                "Can I work remotely?",
                "Who is the CEO?" # Out of scope test
            ]
            st.write("Test Questions:")
            for q in test_queries:
                if st.button(f"▶️ {q}", key=q):
                    st.session_state.query_input = q

    # CHAT INTERFACE
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display Category
            if "category" in message:
                st.markdown(f'<span class="category-tag">{message["category"]}</span>', unsafe_allow_html=True)
            
            st.markdown(message["content"])
            
            # Display Sources with Scores (Feature 1)
            if "sources" in message:
                with st.expander("🔍 View Retrieval Details & Sources"):
                    for i, (doc, score) in enumerate(message["sources"]):
                        st.markdown(f"**Document {i+1}** | Distance: `{score:.4f}` *(Lower is better)*")
                        st.caption(doc.page_content[:200] + "...")

    # User Input Handling
    # Check if Eval mode injected a query
    if 'query_input' in st.session_state:
        prompt = st.session_state.query_input
        del st.session_state.query_input
    else:
        prompt = st.chat_input("Ask a question...")

    if prompt:
        if 'api_key' not in st.session_state:
            st.error("Please enter API Key in sidebar.")
        else:
            # User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, sources, category = user_input(prompt)
                        
                        # Display
                        st.markdown(f'<span class="category-tag">{category}</span>', unsafe_allow_html=True)
                        st.markdown(answer)
                        
                        # Sources Expander
                        if sources:
                            with st.expander("🔍 View Retrieval Details & Sources"):
                                for i, (doc, score) in enumerate(sources):
                                    st.markdown(f"**Document {i+1}** | Distance: `{score:.4f}`")
                                    st.caption(doc.page_content[:200] + "...")

                        # Save History
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "category": category,
                            "sources": sources
                        })

                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()