# 🧠 DocuGPT

Ask any question about a PDF — powered by GPT-4 and vector search.

## Features
- Upload PDFs
- Ask natural language questions
- Accurate answers using local embeddings + GPT
- Local, private, and expandable

## Setup

```bash
git clone https://github.com/yourusername/docuGPT.git
cd docuGPT
pip install -r requirements.txt
touch .env  # add your OPENAI_API_KEY
streamlit run app.py
