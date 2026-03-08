import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------
# Page UI
# -------------------------

st.title("Meeting Intelligence RAG System")
st.write("Upload a meeting transcript and ask questions about it.")


# -------------------------
# Load Models
# -------------------------

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


# -------------------------
# File Upload
# -------------------------

uploaded_file = st.file_uploader("Upload Transcript", type=["txt"])


if uploaded_file:

    text = uploaded_file.read().decode("utf-8")

    st.subheader("Transcript Preview")
    st.write(text[:1000])


    # -------------------------
    # Chunking
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)


    # -------------------------
    # Load Models
    # -------------------------

    embed_model = load_embedding_model()
    tokenizer, llm_model = load_llm()


    # -------------------------
    # Embeddings + FAISS
    # -------------------------

    embeddings = embed_model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))


    # -------------------------
    # Meeting Summary
    # -------------------------

    st.subheader("Meeting Summary")

    summary_prompt = f"""
Summarize the following meeting discussion clearly.

Transcript:
{text}

Summary:
"""

    inputs = tokenizer(summary_prompt, return_tensors="pt", truncation=True)

    outputs = llm_model.generate(**inputs, max_new_tokens=120)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write(summary)


    # -------------------------
    # Key Decisions
    # -------------------------

    st.subheader("Key Decisions")

    decision_prompt = f"""
From the meeting transcript below, list ONLY the final decisions made by the team.

Transcript:
{text}

Decisions:
"""

    inputs = tokenizer(decision_prompt, return_tensors="pt", truncation=True)

    outputs = llm_model.generate(**inputs, max_new_tokens=80)

    decisions = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write(decisions)


    # -------------------------
    # Action Items
    # -------------------------

    st.subheader("Action Items")

    action_prompt = f"""
From the meeting transcript below, extract the tasks assigned to people.

Format:
Person - Task

Transcript:
{text}

Action Items:
"""

    inputs = tokenizer(action_prompt, return_tensors="pt", truncation=True)

    outputs = llm_model.generate(**inputs, max_new_tokens=80)

    actions = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write(actions)


    # -------------------------
    # Question Answering
    # -------------------------

    question = st.text_input("Ask a question about the meeting")


    if question and question.strip() != "":

        q_embedding = embed_model.encode([question])

        # retrieve more chunks
        distances, indices = index.search(np.array(q_embedding), k=5)

        context = " ".join([chunks[i] for i in indices[0]])

        prompt = f"""
You are an AI assistant helping with meeting notes.

Answer the question using ONLY the context provided.

Context:
{context}

Question:
{question}

Answer clearly:
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = llm_model.generate(**inputs, max_new_tokens=120)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Answer")

        st.write(answer)