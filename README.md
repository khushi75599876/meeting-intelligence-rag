# Meeting Intelligence RAG System

Live Demo : https://meeting-intelligence-rag-bvamefydtcozp6sow7zgqf.streamlit.app/

This project is a **RAG-based AI system** that analyzes meeting transcripts and automatically generates:
* Meeting summaries
* Key decisions
* Action items
* Answers to questions about the meeting

## Tech Stack

* Python
* Streamlit
* FAISS (vector search)
* Sentence Transformers
* HuggingFace Transformers (FLAN-T5)

## Features

* Upload meeting transcript (.txt)
* Automatic meeting summary generation
* Extract decisions and action items
* Ask questions about the meeting using RAG

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit link in your browser.

## Future Improvements

* Cross-meeting memory
* Speaker detection
* Better UI for meeting insights
* Export meeting reports
