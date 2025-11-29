# My RAG Project

## Overview

A Retrieval‑Augmented Generation (RAG) application that combines a **Streamlit** front‑end with a **FastAPI** back‑end to provide intelligent, context‑aware answers over a custom document corpus.

## Features

- 📄 Load and index documents (PDF, TXT, Markdown) using **FAISS**.
- 🤖 Query the indexed knowledge base with **OpenAI** (or any compatible LLM).
- 🎨 Interactive UI built with **Streamlit** for real‑time chat experience.
- ⚡️ Simple Docker setup for reproducible local development.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/my_rag_project.git
cd my_rag_project

# Create a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the backend (FastAPI)
uvicorn backend:app --reload

# In a new terminal, run the Streamlit front‑end
streamlit run app.py
```

Open your browser at `http://localhost:8501` to start chatting with the RAG system.

## Project Structure

```
my_rag_project/
├─ app.py            # Streamlit UI entry point
├─ backend.py        # FastAPI server handling document loading & retrieval
├─ requirements.txt  # Python dependencies
├─ README.md         # This file
└─ docs/             # Optional folder for source documents
```

## Configuration

- **LLM Provider** – Set the `OPENAI_API_KEY` environment variable or modify `backend.py` to use another provider.
- **Document Folder** – Place your source files in the `docs/` directory; the backend will index them on startup.

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. Follow the existing code style and ensure all tests pass.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
