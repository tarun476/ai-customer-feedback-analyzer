# 📊 AI Customer Feedback Dashboard

An interactive NLP web application built with **Streamlit**, **Pandas**, and **Hugging Face Transformers** (`distilbert-base-uncased-finetuned-sst-2-english`) to perform batch sentiment analysis on customer review CSV files.

## Features
- **Batch CSV Upload:** Ingests CSV datasets containing customer reviews.
- **Transformer Inference:** Uses a cached Hugging Face DistilBERT pipeline for fast sentiment classification.
- **Visual Analytics:** Generates an interactive bar chart summarizing sentiment distribution.
- **CSV Export:** Allows users to download the analyzed results and confidence scores.

## Setup & Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
3. Upload `sample_reviews.csv` in the browser interface and click **Run AI Batch Analysis**.
