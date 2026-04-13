import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import io

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Meeting Intelligence RAG", layout="wide")
st.title("Meeting Intelligence RAG System")
st.write("Upload a meeting transcript (TXT, PDF, or PPTX) and get AI-powered insights.")

# -------------------------
# Groq Client
# -------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

MAX_CONTEXT_CHARS = 3000

def ask_groq(prompt):
    if len(prompt) > 4000:
        prompt = prompt[:4000]
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate response. Please try again. (Error: {str(e)[:100]})"

# -------------------------
# File Parsers
# -------------------------
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_pdf(file):
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_pptx(file):
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(file.read()))
        text = ""
        for i, slide in enumerate(prs.slides, 1):
            text += f"\n--- Slide {i} ---\n"
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text += shape.text.strip() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PPTX: {e}")
        return ""

# -------------------------
# Embedding Model
# -------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Get top relevant chunks via FAISS
# -------------------------
def get_top_chunks(query, embed_model, index, chunks, k=3, max_chars=MAX_CONTEXT_CHARS):
    q_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(q_vec), k=min(k, len(chunks)))
    context = "\n\n".join([chunks[i] for i in indices[0]])
    return context[:max_chars]

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Transcript",
    type=["txt", "pdf", "pptx"]
)

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    with st.spinner("Reading file..."):
        if file_type == "txt":
            text = extract_text_from_txt(uploaded_file)
        elif file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "pptx":
            text = extract_text_from_pptx(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    if not text:
        st.error("No text could be extracted. Please check your file.")
        st.stop()

    # -------------------------
    # Chunking + FAISS Index
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_text(text)

    with st.spinner("Building search index..."):
        embed_model = load_embedding_model()
        embeddings = embed_model.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

    # -------------------------
    # Meeting Summary (auto-generated once)
    # -------------------------
    st.subheader("Meeting Summary")
    with st.spinner("Generating summary..."):
        summary_context = get_top_chunks(
            "meeting agenda discussion main topics overview",
            embed_model, index, chunks, k=3
        )
        summary = ask_groq(
            f"You are a meeting analyst. Read the meeting notes below and write a "
            f"clear summary in 3-4 sentences covering the main topics discussed.\n\n"
            f"Meeting notes:\n{summary_context}\n\nSummary:"
        )
    st.info(summary)

    st.divider()

    # -------------------------
    # Q&A Search Bar
    # -------------------------
    st.subheader("Ask a Question About the Meeting")
    question = st.text_input("Type your question and press Enter", placeholder="e.g. Who is responsible for the database fix?")

    if question and question.strip():
        with st.spinner("Finding answer..."):
            context = get_top_chunks(question, embed_model, index, chunks, k=3)
            answer = ask_groq(
                f"You are a helpful meeting assistant. Answer the question below using "
                f"ONLY the provided context. If the answer is not in the context, say "
                f"'This information was not found in the transcript.'\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        st.success(answer)
