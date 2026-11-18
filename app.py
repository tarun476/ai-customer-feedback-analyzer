import streamlit as st
import pandas as pd
from transformers import pipeline

# 1. Page Title & Setup
st.title("📊 AI Customer Feedback Dashboard")
st.write("Upload a CSV of customer reviews to analyze sentiment at scale.")

# 2. Load the AI Model (Cached so it only loads once)
@st.cache_resource
def load_sentiment_model():
    # Using a lightweight, fast Transformer model from Hugging Face
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

with st.spinner("Loading AI Model into memory... Please wait."):
    sentiment_analyzer = load_sentiment_model()

# 3. File Uploader for Batch CSV Processing
uploaded_file = st.file_uploader("Upload CSV file with a column named 'review'", type=["csv"])

if uploaded_file is not None:
    # Read CSV using pandas (similar to reading data tables)
    df = pd.read_csv(uploaded_file)
    
    if "review" not in df.columns:
        st.error("Your CSV file must contain a column named 'review'.")
    else:
        st.success(f"Successfully loaded {len(df)} reviews!")
        
        if st.button("Run AI Batch Analysis"):
            results = []
            
            # Loop through reviews and run AI inference
            with st.spinner("Analyzing sentiments..."):
                for text in df["review"].astype(str):
                    prediction = sentiment_analyzer(text)[0]
                    results.append({
                        "review": text,
                        "sentiment": prediction["label"],
                        "confidence": round(prediction["score"] * 100, 2)
                    })
            
            # Convert results into a new DataFrame
            result_df = pd.DataFrame(results)
            
            # 4. Display Analytics & Charts
            st.subheader("Analysis Summary")
            sentiment_counts = result_df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)
            
            # Show detailed table
            st.subheader("Detailed Results")
            st.dataframe(result_df)
            
            # Download button for processed data
            csv_export = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_export,
                file_name="analyzed_sentiment_results.csv",
                mime="text/csv",
            )
