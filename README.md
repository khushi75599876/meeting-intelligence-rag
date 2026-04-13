# Meeting Intelligence RAG System

**Live Demo:** [meeting-intelligence-rag-bvamefydtcozp6sow7zgqf.streamlit.app](https://meeting-intelligence-rag-bvamefydtcozp6sow7zgqf.streamlit.app)

A RAG-based AI system that analyzes meeting transcripts — upload your file and instantly get a summary plus a search bar to ask anything about the meeting.

---

## Features

- Upload meeting transcript (`.txt`, `.pdf`, `.pptx`)
- Automatic meeting summary generation
- Ask any question about the meeting using RAG-powered search

---

## Tech Stack

- **Python**
- **Streamlit** — UI
- **FAISS** — vector similarity search
- **Sentence Transformers** — `all-MiniLM-L6-v2` for embeddings
- **Groq API** — `llama-3.1-8b-instant` for fast LLM inference
- **LangChain Text Splitters** — document chunking

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your Groq API key to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Then open the Streamlit link in your browser.

---

## Future Improvements

- Cross-meeting memory
- Speaker detection
- Export meeting reports
- Better UI for meeting insights
